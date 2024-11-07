#!/usr/bin/env python3
#
# Created on Wed Jan 22 2025 10:28:28
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from multimethod import multimethod

from usf.utils.files import read_file
from usf.utils.params import (
    BINARY_DTYPE_CODE_TO_NUMPY,
    NUMPY_DTYPE_TO_BINARY_CODE,
    SPHERICAL_BIN_HEADER_SIZE,
    SPHERICAL_BIN_MAGIC,
    SPHERICAL_BIN_VERSION,
)
from usf.utils.spherical import (
    cartesian2polar,
    colorize_locations,
    normalize_cartesian,
    normalize_polar,
    polar2cartesian,
)
from usf.utils.torch_numpy import (
    allclose,
    array_size_MiB,
    assert_batch_allclose,
    concatenate,
    copy_or_clone,
    split,
    stack,
    to_numpy,
    to_torch,
)


def _normalize_spherical_save_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a destination path for :meth:`SphericalImage.save` / :meth:`BatchSphericalImage.save`.

    :func:`numpy.savez` appends the ``.npz`` extension when the given path has **no**
    suffix. This project supports both ``.npz`` and ``.bin``; if *path* has no suffix,
    ``.npz`` is appended (same default as the usual compressed-arrays case). If *path*
    already has a suffix, it must be exactly ``.npz`` or ``.bin``.

    Args:
        path (str | os.PathLike[str]): Intended output file path before normalization.

    Returns:
        pathlib.Path: Absolute or relative path object whose suffix is ``.npz`` or ``.bin``.

    Raises:
        ValueError: If *path* has a suffix other than ``.npz`` or ``.bin``.
    """
    p = Path(path)
    suf = p.suffix.lower()
    if suf == "":
        return p.with_suffix(".npz")
    if suf in (".npz", ".bin"):
        return p
    raise ValueError(
        "Spherical save path must end with '.npz' or '.bin', or have no suffix "
        "(then '.npz' is appended, like numpy.savez); "
        f"got suffix {suf!r}."
    )


def _encode_batch_spherical_bin(
    batch_value: np.ndarray,
    vector: np.ndarray,
    polar: np.ndarray,
    mask: np.ndarray,
) -> bytes:
    """Serialize batch spherical arrays to USF ``.bin`` bytes (little-endian, C-contiguous).

    Wire format: 64-byte header (constants in ``usf.utils.params``; ``N``, ``B``, ``C``, four
    dtype codes), then payloads for ``batch_value``, ``vector``, ``polar``, ``mask``.

    Args:
        batch_value (np.ndarray): Per-batch pixel data of shape ``(B, N, C)``; dtype must be
            mappable via ``NUMPY_DTYPE_TO_BINARY_CODE`` in ``usf.utils.params``.
        vector (np.ndarray): Unit Cartesian directions per pixel, shape ``(N, 3)``; dtype as above.
        polar (np.ndarray): Spherical coordinates ``(theta, phi)`` per pixel, shape ``(N, 2)``;
            dtype as above.
        mask (np.ndarray): Boolean (or compatible) mask per pixel, shape ``(N,)``; dtype as above.

    Returns:
        bytes: Full on-disk contents (header plus four array blobs).

    Raises:
        TypeError: If any array dtype is not listed in ``NUMPY_DTYPE_TO_BINARY_CODE`` in
            ``usf.utils.params``.
    """

    def _le(a: np.ndarray) -> bytes:
        dt = np.dtype(a.dtype).newbyteorder("<")
        return np.ascontiguousarray(a, dtype=dt).tobytes()

    b, n, c = batch_value.shape
    codes: list[int] = []
    for name, arr in (
        ("batch_value", batch_value),
        ("vector", vector),
        ("polar", polar),
        ("mask", mask),
    ):
        dt = np.dtype(arr.dtype)
        if dt not in NUMPY_DTYPE_TO_BINARY_CODE:
            raise TypeError(
                f"Unsupported dtype {dt!r} for {name} in .bin export; extend "
                "NUMPY_DTYPE_TO_BINARY_CODE in usf.utils.params"
            )
        codes.append(NUMPY_DTYPE_TO_BINARY_CODE[dt])
    d_bv, d_v, d_p, d_m = codes[0], codes[1], codes[2], codes[3]

    header = bytearray(SPHERICAL_BIN_HEADER_SIZE)
    header[0:8] = SPHERICAL_BIN_MAGIC
    struct.pack_into("<II", header, 8, SPHERICAL_BIN_VERSION, 0)
    struct.pack_into("<Q", header, 16, n)
    struct.pack_into("<II", header, 24, b, c)
    struct.pack_into("<BBBB", header, 32, d_bv, d_v, d_p, d_m)
    return bytes(header) + _le(batch_value) + _le(vector) + _le(polar) + _le(mask)


def _decode_batch_spherical_bin(
    data: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse USF ``.bin`` bytes into ``batch_value``, ``vector``, ``polar``, ``mask`` copies.

    Args:
        data (bytes): Complete file bytes produced by :func:`_encode_batch_spherical_bin` (or a
            compatible writer using the same wire format).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``(batch_value, vector, polar, mask)``
            where ``batch_value`` has shape ``(B, N, C)``, ``vector`` ``(N, 3)``, ``polar`` ``(N, 2)``,
            ``mask`` ``(N,)``; each array is an owning copy (not a view into *data*).

    Raises:
        ValueError: If *data* is truncated, has wrong magic, unsupported version, unknown
            dtype code, or length mismatch.
    """
    if len(data) < SPHERICAL_BIN_HEADER_SIZE:
        raise ValueError("File too small for spherical .bin header.")
    if data[0:8] != SPHERICAL_BIN_MAGIC:
        raise ValueError(
            f"Bad magic: expected {SPHERICAL_BIN_MAGIC!r}, got {data[0:8]!r}."
        )
    version, _rsv = struct.unpack_from("<II", data, 8)
    if version != SPHERICAL_BIN_VERSION:
        raise ValueError(
            f"Unsupported spherical .bin version {version} (expected {SPHERICAL_BIN_VERSION})."
        )
    n = struct.unpack_from("<Q", data, 16)[0]
    b, c = struct.unpack_from("<II", data, 24)
    d_bv, d_v, d_p, d_m = struct.unpack_from("<BBBB", data, 32)
    for code in (d_bv, d_v, d_p, d_m):
        if code not in BINARY_DTYPE_CODE_TO_NUMPY:
            raise ValueError(f"Unknown dtype code {code} in spherical .bin.")

    es_bv = np.dtype(BINARY_DTYPE_CODE_TO_NUMPY[d_bv]).itemsize
    es_v = np.dtype(BINARY_DTYPE_CODE_TO_NUMPY[d_v]).itemsize
    es_p = np.dtype(BINARY_DTYPE_CODE_TO_NUMPY[d_p]).itemsize
    es_m = np.dtype(BINARY_DTYPE_CODE_TO_NUMPY[d_m]).itemsize
    pay = b * n * c * es_bv + n * 3 * es_v + n * 2 * es_p + n * es_m
    if len(data) != SPHERICAL_BIN_HEADER_SIZE + pay:
        raise ValueError(
            f"File size mismatch: expected {SPHERICAL_BIN_HEADER_SIZE + pay} bytes, got {len(data)}."
        )
    off = SPHERICAL_BIN_HEADER_SIZE
    dt_bv = BINARY_DTYPE_CODE_TO_NUMPY[d_bv].newbyteorder("<")
    dt_v = BINARY_DTYPE_CODE_TO_NUMPY[d_v].newbyteorder("<")
    dt_p = BINARY_DTYPE_CODE_TO_NUMPY[d_p].newbyteorder("<")
    dt_m = BINARY_DTYPE_CODE_TO_NUMPY[d_m].newbyteorder("<")
    nb_bv = b * n * c * dt_bv.itemsize
    nb_v = n * 3 * dt_v.itemsize
    batch_value = np.frombuffer(data, dtype=dt_bv, count=b * n * c, offset=off).reshape(
        b, n, c
    )
    off += nb_bv
    vector = np.frombuffer(data, dtype=dt_v, count=n * 3, offset=off).reshape(n, 3)
    off += nb_v
    polar = np.frombuffer(data, dtype=dt_p, count=n * 2, offset=off).reshape(n, 2)
    off += n * 2 * dt_p.itemsize
    mask = np.frombuffer(data, dtype=dt_m, count=n, offset=off).reshape(n)
    return (
        batch_value.copy(),
        vector.copy(),
        polar.copy(),
        mask.copy(),
    )


@dataclass
class SphericalImage:
    """
    Spherical Image Class to store per-pixel data (`value`) along with pixel locations in both Cartesian (`vector`) and spherical (`polar`) representations.

    The class enforces that at least one of `vector` or `polar` is provided. The other is auto-computed in __post_init__.
    If value is not provided, the spherical image locations are colorized with default colors.
    """

    value: np.ndarray | torch.Tensor | None = field(default=None)
    vector: np.ndarray | torch.Tensor | None = field(default=None)
    polar: np.ndarray | torch.Tensor | None = field(default=None)
    mask: np.ndarray | torch.Tensor | None = field(default=None)

    @multimethod
    def __init__(  # type: ignore
        self,
        value: np.ndarray | torch.Tensor | None = None,
        vector: np.ndarray | torch.Tensor | None = None,
        polar: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
    ):
        """Default initialization method

        Args:
            value (np.ndarray | torch.Tensor | None): array or tensor of shape (N, C) or (N, ...) containing pixel data (e.g. RGB, depth, etc.). Defaults to None.
            vector (np.ndarray | torch.Tensor | None): array or tensor of shape (N, 3) for unit Cartesian coordinates. Defaults to None.
            polar (np.ndarray | torch.Tensor | None): array or tensor of shape (N, 2) for spherical coords (theta, phi) in radians, i.e. latitude, longitude. Defaults to None.
            mask (np.ndarray | torch.Tensor | None): array or tensor of shape (N,) for boolean mask. Defaults to None.
        """
        self.value = value
        self.vector = vector
        self.polar = polar
        self.mask = mask

        self.__post_init__()

    @multimethod
    def __init__(self, path: str | Path):
        """Load spherical image from ``.npz`` or ``.bin`` (``B=1`` batch layout).

        Args:
            path (str | Path): Filesystem path to a ``.npz`` (legacy keys) or ``.bin``
                (USF batch wire format with ``B=1``). For other :class:`os.PathLike` values, wrap
                with ``Path(...)`` (``multimethod`` does not dispatch on arbitrary pathlikes here).
        """
        self._init_from_spherical_file_path(Path(path))

    def _init_from_spherical_file_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            loaded = read_file(os.fspath(path))
            assert "value" in loaded, f"Input file {path} is not a spherical image"

            self.value = loaded["value"]
            self.vector = loaded["vector"] if "vector" in loaded else None
            self.polar = loaded["polar"] if "polar" in loaded else None
            self.mask = loaded["mask"] if "mask" in loaded else None
        elif suffix == ".bin":
            batch_value, vector, polar, mask = _decode_batch_spherical_bin(
                path.read_bytes()
            )
            if batch_value.shape[0] != 1:
                raise ValueError(
                    f"SphericalImage requires batch size B=1 in .bin, got B={batch_value.shape[0]}; "
                    "use BatchSphericalImage for multi-frame data."
                )
            self.value = batch_value[0]
            self.vector = vector
            self.polar = polar
            self.mask = mask
        else:
            raise ValueError(
                f"Unsupported spherical image path suffix {suffix!r}; use '.npz' or '.bin'."
            )

        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.vector is None and self.polar is None:
            raise ValueError(
                "Must provide at least one of `vector` or `polar` for spherical image locations."
            )

        # ! Not checking consistency between vector and polar when they are both provided

        # If only polar is given, compute vector.
        if self.vector is None:
            # shape check => polar must be (...,2)
            if self.polar.shape[-1] != 2:
                raise ValueError("`polar` must have last dimension=2 for (theta, phi).")
            self.polar = normalize_polar(self.polar)
            self.vector = polar2cartesian(self.polar)

        # If only vector is given, compute polar.
        if self.polar is None:
            # shape check => vector must be (...,3)
            if self.vector.shape[-1] != 3:
                raise ValueError("`vector` must have last dimension=3 for (x, y, z).")
            self.polar = cartesian2polar(self.vector)

        # Flat the dimensions of vector and polar except the last one.
        self.vector = self.vector.reshape(-1, 3)
        self.polar = self.polar.reshape(-1, 2)

        # Colorize the spherical image locations with default color if value is not provided
        if self.value is None:
            self.value = colorize_locations(self.polar)
        else:
            # Flatten the existing value array/tensor
            self.value = self.value.reshape(-1, self.value.shape[-1])

        if self.mask is None:
            if isinstance(self.value, torch.Tensor):
                self.mask = torch.ones(
                    (self.vector.shape[0],), dtype=torch.bool, device=self.vector.device
                )
            else:
                self.mask = np.ones((self.vector.shape[0],), dtype=bool)
        self.mask = self.mask.reshape(-1)

        # Ensure shapes line up.
        n_value = self.value.shape[0]
        n_vector = self.vector.shape[0]
        n_polar = self.polar.shape[0]
        n_mask = self.mask.shape[0]
        assert (
            n_value == n_vector == n_polar == n_mask
        ), f"Mismatch in number of pixels: value.shape[0]={n_value}, vector.shape[0]={n_vector}, polar.shape[0]={n_polar}, mask.shape[0]={n_mask}."

        # ensure on the same device if provided as tensors
        if isinstance(self.value, torch.Tensor):
            target_device = self.value.device
            self.vector = self.vector.to(target_device)
            self.polar = self.polar.to(target_device)
            self.mask = self.mask.to(target_device)

        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """safe setattr for SphericalImage

        Args:
            name (str): name of the attribute to set
            value (Any): value to set
        """
        # during init or for private fields, just set
        if not getattr(self, "_initialized", False) or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name in {"value", "vector", "polar", "mask"}:
            assert value is not None, f"{name} cannot be set to None"

        # ----- geometry touched: auto-complete counterpart -----
        if name == "vector":
            assert type(value) is type(
                self.vector
            ), "vector must have the same backend as spherical image"
            assert value.shape[-1] == 3, "vector must have shape (..., 3)"
            vector = normalize_cartesian(value.reshape(-1, 3))
            polar = cartesian2polar(vector)
            assert (
                vector.shape[0] == self.vector.shape[0]
            ), f"vector.shape[0]={vector.shape[0]} != self.vector.shape[0]={self.vector.shape[0]}"
            if isinstance(vector, torch.Tensor):
                # enforce same device
                vector = vector.to(device=self.vector.device)
                polar = polar.to(device=self.polar.device)
            object.__setattr__(self, "vector", vector)
            object.__setattr__(self, "polar", polar)

        elif name == "polar":
            assert type(value) is type(
                self.polar
            ), "polar must have the same backend as spherical image"
            assert value.shape[-1] == 2, "polar must have shape (..., 2)"
            polar = normalize_polar(value.reshape(-1, 2))
            vector = polar2cartesian(polar)
            assert (
                polar.shape[0] == self.polar.shape[0]
            ), f"polar.shape[0]={polar.shape[0]} != self.polar.shape[0]={self.polar.shape[0]}"
            if isinstance(polar, torch.Tensor):
                # enforce same device
                vector = vector.to(device=self.vector.device)
                polar = polar.to(device=self.polar.device)
            object.__setattr__(self, "vector", vector)
            object.__setattr__(self, "polar", polar)

        # ----- value/mask touched: reshape & check only -----
        elif name == "value":
            assert type(value) is type(
                self.value
            ), "value must have the same backend as spherical image"
            value = value.reshape(-1, value.shape[-1])
            assert (
                value.shape[0] == self.value.shape[0]
            ), f"value.shape[0]={value.shape[0]} != self.value.shape[0]={self.value.shape[0]}"
            if isinstance(value, torch.Tensor):
                # enforce same device
                value = value.to(device=self.value.device)
            object.__setattr__(self, "value", value)

        elif name == "mask":
            assert type(value) is type(
                self.mask
            ), "mask must have the same backend as spherical image"
            mask = value.reshape(-1)
            assert (
                mask.shape[0] == self.mask.shape[0]
            ), f"mask.shape[0]={mask.shape[0]} != self.mask.shape[0]={self.mask.shape[0]}"
            # enforce boolean dtype and same device
            if isinstance(mask, torch.Tensor):
                mask = mask.to(dtype=torch.bool, device=self.mask.device)
            else:
                mask = mask.astype(bool)
            object.__setattr__(self, "mask", mask)

        else:
            object.__setattr__(self, name, value)

    def apply_mask(self) -> SphericalImage:
        """
        Return a new SphericalImage where all fields are filtered using the current mask.
        The resulting image will have mask set to all True.
        """
        if isinstance(self.value, torch.Tensor):
            assert (
                self.value.device
                == self.vector.device
                == self.polar.device
                == self.mask.device
            ), f"Device mismatch: value.device={self.value.device} != vector.device={self.vector.device} != polar.device={self.polar.device} != mask.device={self.mask.device}"

        new_value = self.value[self.mask]
        new_vector = self.vector[self.mask]
        new_polar = self.polar[self.mask]

        return SphericalImage(
            value=new_value,
            vector=new_vector,
            polar=new_polar,
        )

    def clone(self) -> SphericalImage:
        """
        Make a deep-ish copy of the data.
        (You could also rely on 'copy.deepcopy(self)' if desired.)
        """
        return SphericalImage(
            value=copy_or_clone(self.value),
            vector=copy_or_clone(self.vector),
            polar=copy_or_clone(self.polar),
            mask=copy_or_clone(self.mask),
        )

    def copy(self) -> SphericalImage:
        """Create a deep copy of this SphericalImage (alias for ``clone``).

        Returns:
            SphericalImage: Independent copy with cloned/copied arrays.
        """
        return SphericalImage(
            value=copy_or_clone(self.value),
            vector=copy_or_clone(self.vector),
            polar=copy_or_clone(self.polar),
            mask=copy_or_clone(self.mask),
        )

    def to_torch(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> SphericalImage:
        """Convert all fields to ``torch.Tensor`` on the given device/dtype.

        Args:
            device (torch.device | None, optional): Target device. Defaults to None.
            dtype (torch.dtype | None, optional): Target dtype for value/vector/polar.
                The mask is always cast to ``torch.bool``. Defaults to None.

        Returns:
            SphericalImage: New instance with all fields as torch tensors.
        """
        return SphericalImage(
            value=to_torch(self.value, device=device, dtype=dtype),
            vector=to_torch(self.vector, device=device, dtype=dtype),
            polar=to_torch(self.polar, device=device, dtype=dtype),
            mask=to_torch(self.mask, device=device, dtype=torch.bool),
        )

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> SphericalImage:
        """Alias for ``to_torch`` — move/cast all fields.

        Args:
            device (torch.device | None, optional): Target device. Defaults to None.
            dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

        Returns:
            SphericalImage: New instance on the target device/dtype.
        """
        return SphericalImage(
            value=to_torch(self.value, device=device, dtype=dtype),
            vector=to_torch(self.vector, device=device, dtype=dtype),
            polar=to_torch(self.polar, device=device, dtype=dtype),
            mask=to_torch(self.mask, device=device, dtype=torch.bool),
        )

    def to_numpy(self) -> SphericalImage:
        """
        Convert the data to numpy.ndarray.
        """
        return SphericalImage(
            value=to_numpy(self.value),
            vector=to_numpy(self.vector),
            polar=to_numpy(self.polar),
            mask=to_numpy(self.mask),
        )

    def save(self, path: str | os.PathLike[str]) -> SphericalImage:
        """Save as ``.npz`` or ``.bin`` (batch layout with ``B=1``).

        If *path* has no suffix, ``.npz`` is appended (same default as :func:`numpy.savez`).
        Otherwise the suffix must be ``.npz`` or ``.bin``.

        Args:
            path (str | os.PathLike[str]): Destination file path; normalized by
                :func:`_normalize_spherical_save_path`.

        Returns:
            SphericalImage: ``self`` for chaining; underlying arrays are not modified.

        Raises:
            ValueError: If *path* has a suffix other than ``.npz`` or ``.bin``.
        """
        p = _normalize_spherical_save_path(path)
        suffix = p.suffix.lower()
        np_spherical_image = self.to_numpy()

        if suffix == ".npz":
            np.savez(
                os.fspath(p),
                value=np_spherical_image.value,
                vector=np_spherical_image.vector,
                polar=np_spherical_image.polar,
                mask=np_spherical_image.mask,
            )
        else:
            bsi = BatchSphericalImage(np_spherical_image)
            np_b = bsi.to_numpy()
            p.write_bytes(
                _encode_batch_spherical_bin(
                    np_b.batch_value,
                    np_b.vector,
                    np_b.polar,
                    np_b.mask,
                )
            )

        return self

    @multimethod
    def split(self, indices_or_sections: int | list[int]) -> list[SphericalImage]:
        """split the spherical image by value channel

        Args:
            indices_or_sections (int | list[int]): refer to numpy/torch's split

        Returns:
            list[SphericalImage]: list of splitted spherical images
        """

        return [
            SphericalImage(
                value=value,
                vector=self.vector,
                polar=self.polar,
                mask=self.mask,
            )
            for value in split(self.value, indices_or_sections, -1)
        ]

    @property
    def num_pixels(self) -> int:
        """Number of pixels in this spherical image."""
        return self.value.shape[0]

    def __sizeof__(self) -> int:
        """Return the size of this object in Byte

        Returns:
            int: Byte
        """
        size = int(
            (
                array_size_MiB(self.value)
                + array_size_MiB(self.vector)
                + array_size_MiB(self.polar)
                + array_size_MiB(self.mask)
            )
            * 1024**2
        )

        return size

    def __str__(self) -> str:
        """Human-readable summary: backend, dtype, shape, and memory footprint.

        Returns:
            str: Multi-line summary string.
        """
        N, C = self.value.shape

        backend_str = f"{type(self.value).__module__}.{type(self.value).__name__}"

        type_str = f"{self.value.dtype}"

        device_str = (
            f", on {self.value.device}" if isinstance(self.value, torch.Tensor) else ""
        )

        size_mb = round(self.__sizeof__() / (1024**2), 3)

        return (
            "<"
            + backend_str
            + f" SphericalImage with {N} "
            + type_str
            + f" pixels and {C} channels"
            + device_str
            + f" occupying {size_mb} MiB>"
        )

    def __len__(self) -> int:
        """Number of 'pixels' in this spherical image."""
        return self.value.shape[0]

    def __eq__(self, other: object) -> bool:
        """Check if two SphericalImage objects are equal."""
        if not isinstance(other, SphericalImage):
            return NotImplemented
        return (
            allclose(self.value, other.value)
            and allclose(self.vector, other.vector)
            and allclose(self.polar, other.polar)
            and allclose(self.mask, other.mask)
        )

    def __add__(self, other: object) -> SphericalImage:
        """
        Combine two SphericalImage objects by concatenating
        their data along the pixel dimension (axis=0).

        Requirements/Assumptions:
        - Both images must have the same 'channel' dimension in self.value.
        - No duplicates are removed. It's just a naive stacking.
        """
        if not isinstance(other, SphericalImage):
            raise TypeError("Can only add SphericalImage to SphericalImage.")

        # Optional check that the 'channel' dimension (value.shape[1]) matches:
        if self.value.shape[1] != other.value.shape[1]:
            raise ValueError(
                f"Channel mismatch: self.value.shape[1]={self.value.shape[1]} != "
                f"other.value.shape[1]={other.value.shape[1]}"
            )

        if isinstance(self.value, np.ndarray):
            new_value = np.vstack([self.value, other.value])
            new_vector = np.vstack([self.vector, other.vector])
            new_polar = np.vstack([self.polar, other.polar])
            new_mask = np.hstack([self.mask, other.mask])
        elif isinstance(self.value, torch.Tensor):
            new_value = torch.vstack([self.value, other.value])
            new_vector = torch.vstack([self.vector, other.vector])
            new_polar = torch.vstack([self.polar, other.polar])
            new_mask = torch.hstack([self.mask, other.mask])
        else:
            raise TypeError("Unsupported tensor type in SphericalImage")

        return SphericalImage(
            value=new_value,
            vector=new_vector,
            polar=new_polar,
            mask=new_mask,
        )

    def __getitem__(self, index: Any) -> SphericalImage:
        """
        Retrieve a single SphericalImage or a sub-BatchSphericalImage using any indexing such as integer, slice, fancy indexing, etc.

        Args:
            index (Any): Any indexing such as integer, slice, fancy indexing, etc.

        Returns:
            SphericalImage: A SphericalImage corresponding to the selected indices
        """
        return SphericalImage(
            value=self.value[index],
            vector=self.vector[index],
            polar=self.polar[index],
            mask=self.mask[index],
        )


@dataclass
class BatchSphericalImage:
    """
    Spherical Image Class to store per-pixel data (`batch_value`) along with pixel locations in both Cartesian (`vector`)
    and spherical (`polar`) representations.

    The class enforces that at least one of `vector` or `polar` is provided. The other is auto-computed in __post_init__.
    If batch_value is not provided, the spherical image locations are colorized with default colors.
    """

    batch_value: np.ndarray | torch.Tensor | None = field(default=None)
    vector: np.ndarray | torch.Tensor | None = field(default=None)
    polar: np.ndarray | torch.Tensor | None = field(default=None)
    mask: np.ndarray | torch.Tensor | None = field(default=None)

    @multimethod
    def __init__(  # type: ignore
        self,
        batch_value: (
            SphericalImage
            | np.ndarray
            | torch.Tensor
            | list[SphericalImage | np.ndarray | torch.Tensor]
            | None
        ) = None,
        vector: np.ndarray | torch.Tensor | None = None,
        polar: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
    ):
        """Default initialization method

        Args:
            batch_value (SphericalImage | np.ndarray | torch.Tensor | list[SphericalImage | np.ndarray | torch.Tensor] | None):
                description. Defaults to None.
            vector (np.ndarray | torch.Tensor | None): array or tensor of shape (N, 3) for unit Cartesian coordinates. Defaults to None.
            polar (np.ndarray | torch.Tensor | None): array or tensor of shape (N, 2) for polar coordinates (theta, phi) in radians. Defaults to None.
            mask (np.ndarray | torch.Tensor | None): array or tensor for boolean mask. Defaults to None.

        Ensures:
          - At least one of `vector` or `polar` is provided.
          - The type of batch_value, vector, and polar are consistent
          - The number of pixels in batch_value matches the geometry.
          - If only one of the geometry fields is provided, the other is computed.
          - If batch_value is provided as a list, it is combined into a single np.ndarray or torch.Tensor.
          - If batch_value is provided as a single image (SphericalImage or array), a batch dimension is added.
        """
        self.batch_value = batch_value
        self.vector = vector
        self.polar = polar
        self.mask = mask

        # --- Process batch_value ---
        self._parse_batch_value(self.batch_value)

        self.__post_init__()

    @multimethod
    def __init__(self, path: str | Path):
        """Load batch spherical image from ``.npz`` or ``.bin``.

        Args:
            path (str | Path): Filesystem path to ``.npz`` (``batch_value`` or ``value``)
                or ``.bin`` (USF batch wire format). For other :class:`os.PathLike` values, wrap
                with ``Path(...)`` (``multimethod`` does not dispatch on arbitrary pathlikes here).
        """
        self._init_from_batch_spherical_file_path(Path(path))

    def _init_from_batch_spherical_file_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            loaded = read_file(os.fspath(path))

            assert (
                "batch_value" in loaded or "value" in loaded
            ), f"Input {loaded} is not a spherical image or batch spherical image"

            self.batch_value = (
                loaded["batch_value"]
                if "batch_value" in loaded
                else loaded["value"][np.newaxis, ...]
            )
            self.vector = loaded["vector"] if "vector" in loaded else None
            self.polar = loaded["polar"] if "polar" in loaded else None
            self.mask = loaded["mask"] if "mask" in loaded else None
        elif suffix == ".bin":
            self.batch_value, self.vector, self.polar, self.mask = (
                _decode_batch_spherical_bin(path.read_bytes())
            )
        else:
            raise ValueError(
                f"Unsupported batch spherical image path suffix {suffix!r}; use '.npz' or '.bin'."
            )

        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # --- Process geometry (vector and polar) ---
        # Check that at least one of vector or polar is provided.
        assert (
            self.vector is not None or self.polar is not None
        ), "Must provide at least one of `vector` or `polar` for spherical image locations."

        # If only polar is provided, compute vector.
        if self.vector is None:
            if self.polar.shape[-1] != 2:
                raise ValueError(
                    "`polar` must have last dimension = 2 for (theta, phi)."
                )
            self.polar = normalize_polar(self.polar)
            self.vector = polar2cartesian(self.polar)

        # If only vector is provided, compute polar.
        if self.polar is None:
            if self.vector.shape[-1] != 3:
                raise ValueError("`vector` must have last dimension = 3 for (x, y, z).")
            self.polar = cartesian2polar(self.vector)

        # Flatten geometry to shape (N,3) and (N,2).
        self.vector = self.vector.reshape(-1, 3)
        self.polar = self.polar.reshape(-1, 2)

        # If batch_value is not provided, use default colorization.
        if self.batch_value is None:
            colors = colorize_locations(self.polar)  # expected shape (N, C)

            if isinstance(colors, np.ndarray):
                self.batch_value = np.expand_dims(colors, axis=0)
            else:
                self.batch_value = colors.unsqueeze(0)

        # If mask is not provided, use default mask.
        if self.mask is None:
            if isinstance(self.batch_value, torch.Tensor):
                self.mask = torch.ones(
                    (self.vector.shape[0],), dtype=torch.bool, device=self.vector.device
                )
            else:
                self.mask = np.ones((self.vector.shape[0],), dtype=bool)
        self.mask = self.mask.reshape(-1)

        # Check that the pixel count matches geometry.
        n_value = self.batch_value.shape[1]
        n_vector = self.vector.shape[0]
        n_polar = self.polar.shape[0]
        n_mask = self.mask.shape[0]
        assert (
            n_value == n_vector == n_polar == n_mask
        ), f"Mismatch in number of pixels: batch_value.shape[1]={n_value}, vector.shape[0]={n_vector}, polar.shape[0]={n_polar}, mask.shape[0]={n_mask}."

        # check geometry type matches value type
        assert type(self.batch_value) is type(self.vector) and type(
            self.batch_value
        ) is type(
            self.polar
        ), f"type mismatch between input batch_value(SphericalImage/List/Array): {type(self.batch_value)} and geometry(vector: {type(self.vector)}; polar: {type(self.polar)})"

        # ensure on the same device if provided as tensors
        if isinstance(self.batch_value, torch.Tensor):
            target_device = self.batch_value.device
            self.vector = self.vector.to(target_device)
            self.polar = self.polar.to(target_device)
            self.mask = self.mask.to(target_device)

        object.__setattr__(self, "_initialized", True)

    @multimethod
    def _parse_batch_value(self, batch_value: BatchSphericalImage) -> None:  # type: ignore
        input_batch_spherical_image = batch_value

        self.batch_value = input_batch_spherical_image.batch_value
        self.vector = input_batch_spherical_image.vector
        self.polar = input_batch_spherical_image.polar
        self.mask = input_batch_spherical_image.mask

    @multimethod
    def _parse_batch_value(self, batch_value: SphericalImage) -> None:  # type: ignore
        # Overload for SphericalImage

        # trust SphericalImage's internal safeguarding
        input_spherical_image = batch_value

        self.batch_value = input_spherical_image.value
        self.vector = input_spherical_image.vector
        self.polar = input_spherical_image.polar
        self.mask = input_spherical_image.mask

        # prepend batch dimension
        if isinstance(self.batch_value, np.ndarray):
            self.batch_value = self.batch_value[np.newaxis, ...]
        elif isinstance(self.batch_value, torch.Tensor):
            self.batch_value = self.batch_value.unsqueeze(0)

    @multimethod
    def _parse_batch_value(self, batch_value: np.ndarray) -> None:  # type: ignore
        # Overload for np.ndarray
        if batch_value.ndim == 2:
            self.batch_value = np.expand_dims(batch_value, axis=0)

        if self.batch_value.ndim != 3:
            raise ValueError(
                "batch_value must have 2 dimensions (N, C) or 3 dimensions (B, N, C) when provided as a whole np.ndarray."
            )

    @multimethod
    def _parse_batch_value(self, batch_value: torch.Tensor) -> None:  # type: ignore
        # Overload for torch.Tensor
        if batch_value.ndim == 2:
            self.batch_value = batch_value.unsqueeze(0)

        if self.batch_value.ndim != 3:
            raise ValueError(
                "batch_value must have 2 dimensions (N, C) or 3 dimensions (B, N, C) when provided as a whole torch.Tensor."
            )

    @multimethod
    def _parse_batch_value(self, batch_value: list[SphericalImage]) -> None:  # type: ignore
        # Overload for list of spherical images
        assert all(
            isinstance(item, SphericalImage) for item in batch_value
        ), "batch_value list contains heterogenous data types"

        first_img = batch_value[0]

        # Check consistent value type
        common_type = type(first_img.value)
        assert all(
            isinstance(img.value, common_type) for img in batch_value
        ), "Inconsistent types in SphericalImage list."

        # Extract and stack values
        values = [img.value for img in batch_value]

        # Check consistent value shape
        common_value_shape = values[0].shape
        assert all(
            value.shape == common_value_shape for value in values[1:]
        ), "Inconsistent value shape in input batch_value."

        # Check consistent geometry and mask
        # ! naively comparing vector and polar without ripple sorting since sorting value leads to indifferentiability
        assert_batch_allclose(
            [image.vector for image in batch_value],
            "All spherical images must have consistent vector coordinates",
        )
        assert_batch_allclose(
            [image.polar for image in batch_value],
            "All spherical images must have consistent polar coordinates",
        )
        assert_batch_allclose(
            [image.mask for image in batch_value],
            "All spherical images must have consistent mask",
        )

        self.vector = first_img.vector
        self.polar = first_img.polar
        self.mask = first_img.mask

        # Construct batch_value and mask by stacking
        self.batch_value = stack(values, 0)

    @multimethod
    def _parse_batch_value(self, batch_value: list[BatchSphericalImage]) -> None:  # type: ignore
        # Overload for list of spherical images
        assert all(
            isinstance(item, BatchSphericalImage) for item in batch_value
        ), "batch_value list contains heterogenous data types"

        first_img = batch_value[0]

        # Check consistent value type
        common_type = type(first_img.batch_value)
        assert all(
            isinstance(img.batch_value, common_type) for img in batch_value
        ), "Inconsistent types in BatchSphericalImage list."

        # Extract and stack values
        values = [batch_img.batch_value for batch_img in batch_value]

        # Check consistent value shape
        common_value_shape = values[0].shape[-2:]
        assert all(
            value.shape[-2:] == common_value_shape for value in values[1:]
        ), f"Inconsistent batch_value shape in input batch_value, first has last 2 dim {common_value_shape}."

        # Check consistent geometry and mask
        # ! naively comparing vector and polar without ripple sorting since sorting value leads to indifferentiability
        assert_batch_allclose(
            [image.vector for image in batch_value],
            "All batch spherical images must have consistent vector coordinates",
        )
        assert_batch_allclose(
            [image.polar for image in batch_value],
            "All batch spherical images must have consistent polar coordinates",
        )
        assert_batch_allclose(
            [image.mask for image in batch_value],
            "All batch spherical images must have consistent mask",
        )

        self.vector = first_img.vector
        self.polar = first_img.polar
        self.mask = first_img.mask

        # Construct batch_value and mask by concatenate
        self.batch_value = concatenate(values, 0)

    @multimethod
    def _parse_batch_value(self, batch_value: list[np.ndarray | torch.Tensor]) -> None:  # type: ignore
        # Overload for list of numpy arrays or torch Tensors
        first_value = batch_value[0]

        # Check consistent type
        common_type = type(first_value)
        assert all(
            isinstance(value, common_type) for value in batch_value[1:]
        ), "Inconsistent types in numpy/torch list."

        # Check consistent shape
        common_shape = first_value.shape
        assert all(
            value.shape == common_shape for value in batch_value[1:]
        ), "Inconsistent value shape in input batch_value."

        values = []
        for value in batch_value:
            if value.ndim == 2:  # Already (N, C)
                values.append(value)
            elif value.ndim >= 3:
                values.append(value.reshape(-1, value.shape[-1]))
            else:
                raise ValueError(
                    "Each image in batch_value must have at least 2 dimensions (N, C)."
                )

        # Construct batch_value by stacking
        self.batch_value = stack(values, 0)

    @multimethod
    def _parse_batch_value(self, batch_value: None) -> None:  # type: ignore
        # default colorization
        # can't do it here since vector and polar are unprocessed
        return

    @multimethod
    def _parse_batch_value(self, batch_value: object) -> None:
        # Overload for unsupported types
        raise TypeError(f"Unsupported type for batch_value: {type(batch_value)}")

    def __setattr__(self, name: str, value: Any) -> None:
        """safe setattr for BatchSphericalImage

        Args:
            name (str): name of the attribute to set
            value (Any): value to set
        """
        # during init or for private fields, just set
        if not getattr(self, "_initialized", False) or name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        if name in {"batch_value", "vector", "polar", "mask"}:
            assert value is not None, f"{name} cannot be set to None"

        # ----- geometry touched: auto-complete counterpart -----
        if name == "vector":
            assert type(value) is type(
                self.vector
            ), "vector must have the same backend as spherical image"
            assert value.shape[-1] == 3, "vector must have shape (..., 3)"
            vector = normalize_cartesian(value.reshape(-1, 3))
            polar = cartesian2polar(vector)
            assert (
                vector.shape[0] == self.vector.shape[0]
            ), f"vector.shape[0]={vector.shape[0]} != self.vector.shape[0]={self.vector.shape[0]}"
            if isinstance(vector, torch.Tensor):
                # enforce same device
                vector = vector.to(device=self.vector.device)
                polar = polar.to(device=self.polar.device)
            object.__setattr__(self, "vector", vector)
            object.__setattr__(self, "polar", polar)

        elif name == "polar":
            assert type(value) is type(
                self.polar
            ), "polar must have the same backend as spherical image"
            assert value.shape[-1] == 2, "polar must have shape (..., 2)"
            polar = normalize_polar(value.reshape(-1, 2))
            vector = polar2cartesian(polar)
            assert (
                polar.shape[0] == self.polar.shape[0]
            ), f"polar.shape[0]={polar.shape[0]} != self.polar.shape[0]={self.polar.shape[0]}"
            if isinstance(polar, torch.Tensor):
                # enforce same device
                vector = vector.to(device=self.vector.device)
                polar = polar.to(device=self.polar.device)
            object.__setattr__(self, "vector", vector)
            object.__setattr__(self, "polar", polar)

        # ----- batch_value/mask touched: reshape & check only -----
        elif name == "batch_value":
            assert type(value) is type(
                self.batch_value
            ), "value must have the same backend as spherical image"
            batch_value = value.reshape(-1, self.batch_value.shape[-2], value.shape[-1])
            assert (
                batch_value.shape[-2] == self.batch_value.shape[-2]
            ), f"batch_value.shape[-2]={batch_value.shape[-2]} != self.batch_value.shape[-2]={self.batch_value.shape[-2]}"
            if isinstance(batch_value, torch.Tensor):
                # enforce same device
                batch_value = batch_value.to(device=self.batch_value.device)
            object.__setattr__(self, "batch_value", batch_value)

        elif name == "mask":
            assert type(value) is type(
                self.mask
            ), "mask must have the same backend as spherical image"
            mask = value.reshape(-1)
            assert (
                mask.shape[0] == self.mask.shape[0]
            ), f"mask.shape[0]={mask.shape[0]} != self.mask.shape[0]={self.mask.shape[0]}"
            # enforce boolean dtype and same device
            if isinstance(mask, torch.Tensor):
                mask = mask.to(dtype=torch.bool, device=self.mask.device)
            else:
                mask = mask.astype(bool)
            object.__setattr__(self, "mask", mask)

        else:
            object.__setattr__(self, name, value)

    def apply_mask(self) -> BatchSphericalImage:
        """
        Return a new BatchSphericalImage where all spatial locations (across batch) are filtered using the mask.
        The resulting image will have mask set to all True.
        """
        new_batch_value = self.batch_value[:, self.mask]
        new_vector = self.vector[self.mask]
        new_polar = self.polar[self.mask]

        return BatchSphericalImage(
            batch_value=new_batch_value,
            vector=new_vector,
            polar=new_polar,
        )

    def save(self, path: str | os.PathLike[str]) -> BatchSphericalImage:
        """Save as ``.npz`` or ``.bin``.

        If *path* has no suffix, ``.npz`` is appended (same default as :func:`numpy.savez`).
        Otherwise the suffix must be ``.npz`` or ``.bin``.

        Args:
            path (str | os.PathLike[str]): Destination file path; normalized by
                :func:`_normalize_spherical_save_path`.

        Returns:
            BatchSphericalImage: ``self`` for chaining; underlying arrays are not modified.

        Raises:
            ValueError: If *path* has a suffix other than ``.npz`` or ``.bin``.
        """
        p = _normalize_spherical_save_path(path)
        suffix = p.suffix.lower()
        np_batch_spherical_image = self.to_numpy()

        if suffix == ".npz":
            np.savez(
                os.fspath(p),
                batch_value=np_batch_spherical_image.batch_value,
                vector=np_batch_spherical_image.vector,
                polar=np_batch_spherical_image.polar,
                mask=np_batch_spherical_image.mask,
            )
        else:
            p.write_bytes(
                _encode_batch_spherical_bin(
                    np_batch_spherical_image.batch_value,
                    np_batch_spherical_image.vector,
                    np_batch_spherical_image.polar,
                    np_batch_spherical_image.mask,
                )
            )

        return self

    @multimethod
    def split(self, indices_or_sections: int | list[int]) -> list[BatchSphericalImage]:
        """split the batch spherical image by batch_value channel

        Args:
            indices_or_sections (int | list[int]): refer to numpy/torch's split

        Returns:
            list[BatchSphericalImage]: list of splitted batch spherical images
        """

        return [
            BatchSphericalImage(
                batch_value=batch_value,
                vector=self.vector,
                polar=self.polar,
                mask=self.mask,
            )
            for batch_value in split(self.batch_value, indices_or_sections, -1)
        ]

    @property
    def num_pixels(self) -> int:
        """Number of pixels in this batch spherical image."""
        return self.batch_value.shape[1]

    @property
    def num_images(self) -> int:
        """number of images in this batch spherical image

        Returns:
            int: batch size
        """
        return self.batch_value.shape[0]

    def __len__(self) -> int:
        """
        Returns:
            int: batch_size.
        """
        # batch_value is expected to have shape (B, N, C)
        return self.batch_value.shape[0]

    def __eq__(self, other: object) -> bool:
        """Check if two BatchSphericalImage objects are equal."""
        if not isinstance(other, BatchSphericalImage):
            return NotImplemented
        return (
            allclose(self.batch_value, other.batch_value)
            and allclose(self.vector, other.vector)
            and allclose(self.polar, other.polar)
            and allclose(self.mask, other.mask)
        )

    def __sizeof__(self) -> int:
        """Return the size of this object in Byte

        Returns:
            int: Byte
        """
        return (
            array_size_MiB(self.batch_value)
            + array_size_MiB(self.vector)
            + array_size_MiB(self.polar)
            + array_size_MiB(self.mask)
        ) * 1024**2

    def __str__(self) -> str:
        """
        Returns:
            str: String representation showing batch size, number of pixels per image, and number of channels.
        """
        B, N, C = self.batch_value.shape

        backend_str = (
            f"{type(self.batch_value).__module__}.{type(self.batch_value).__name__}"
        )

        type_str = f"{self.batch_value.dtype}"

        device_str = (
            f", on {self.batch_value.device}"
            if isinstance(self.batch_value, torch.Tensor)
            else ""
        )

        size_mb = round(self.__sizeof__() / (1024**2), 3)

        valid_percentage = round((self.mask.float().mean().item()) * 100, 3)

        return (
            "<"
            + backend_str
            + f" BatchSphericalImage with {B} images, {N} "
            + f"{valid_percentage}% valid "
            + type_str
            + f" pixels, and {C} channels"
            + device_str
            + f" occupying {size_mb} MiB>"
        )

    def __add__(
        self, other: BatchSphericalImage | SphericalImage
    ) -> BatchSphericalImage:
        """
        Concatenate two BatchSphericalImage objects or a BatchSphericalImage with a SphericalImage along the batch dimension.

        Args:
            other (BatchSphericalImage | SphericalImage): The image(s) to add.

        Returns:
            BatchSphericalImage: A new BatchSphericalImage with concatenated batch_value.
        """
        # Construct a BatchSphericalImage if the other object is a single SphericalImage
        if isinstance(other, SphericalImage):
            other = BatchSphericalImage(batch_value=other)

        # Check type consistency
        assert type(self.batch_value) is type(
            other.batch_value
        ), f"Inconsistent backend, first is {type(self.batch_value)}, second is {type(other.batch_value)}"

        # Check geometry and mask consistency.
        assert allclose(
            self.vector, other.vector
        ), "Vector coordinates of the batch spherical images do not match."
        assert allclose(
            self.polar, other.polar
        ), "Polar coordinates of the batch spherical images do not match."
        assert allclose(
            self.mask, other.mask
        ), "Mask of the batch spherical images do not match."

        # Concatenate batch_value along the batch dimension.
        new_batch_value = concatenate([self.batch_value, other.batch_value], 0)

        return BatchSphericalImage(
            batch_value=new_batch_value,
            vector=self.vector,
            polar=self.polar,
            mask=self.mask,
        )

    def __getitem__(self, index: int | Any) -> SphericalImage | BatchSphericalImage:
        """
        Retrieve a single SphericalImage or a sub-BatchSphericalImage using an index or slice.

        Args:
            index (int | Any): Index or slice of the batch.

        Returns:
            SphericalImage | BatchSphericalImage: A SphericalImage corresponding to the selected batch element(s).
            Or a BatchSphericalImage if input index is a slice
        """
        selected_value = self.batch_value[index]

        if isinstance(index, int):
            return SphericalImage(
                value=selected_value,
                vector=self.vector,
                polar=self.polar,
                mask=self.mask,
            )
        elif isinstance(index, slice):
            return BatchSphericalImage(
                batch_value=selected_value,
                vector=self.vector,
                polar=self.polar,
                mask=self.mask,
            )
        else:
            raise TypeError(
                f"Indexing with {type(index)} is not supported. Use int or slice."
            )

    def __iter__(self) -> Iterator[SphericalImage]:
        """Generates a single SphericalImage

        Yields:
            SphericalImage: Individual SphericalImage from the batch.
        """
        for i in range(len(self)):
            yield self[i]

    def clone(self) -> BatchSphericalImage:
        """
        Make a deep-ish copy of the data.
        """
        return BatchSphericalImage(
            batch_value=copy_or_clone(self.batch_value),
            vector=copy_or_clone(self.vector),
            polar=copy_or_clone(self.polar),
            mask=copy_or_clone(self.mask),
        )

    def copy(self) -> BatchSphericalImage:
        """Create a deep copy of this BatchSphericalImage (alias for ``clone``).

        Returns:
            BatchSphericalImage: Independent copy with cloned/copied arrays.
        """
        return BatchSphericalImage(
            batch_value=copy_or_clone(self.batch_value),
            vector=copy_or_clone(self.vector),
            polar=copy_or_clone(self.polar),
            mask=copy_or_clone(self.mask),
        )

    def to_torch(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> BatchSphericalImage:
        """Convert all fields to ``torch.Tensor`` on the given device/dtype.

        Args:
            device (torch.device | None, optional): Target device. Defaults to None.
            dtype (torch.dtype | None, optional): Target dtype for batch_value/vector/polar.
                The mask is always cast to ``torch.bool``. Defaults to None.

        Returns:
            BatchSphericalImage: New instance with all fields as torch tensors.
        """
        return BatchSphericalImage(
            batch_value=to_torch(self.batch_value, device=device, dtype=dtype),
            vector=to_torch(self.vector, device=device, dtype=dtype),
            polar=to_torch(self.polar, device=device, dtype=dtype),
            mask=to_torch(self.mask, device=device, dtype=torch.bool),
        )

    def to(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> BatchSphericalImage:
        """Alias for ``to_torch`` — move/cast all fields.

        Args:
            device (torch.device | None, optional): Target device. Defaults to None.
            dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

        Returns:
            BatchSphericalImage: New instance on the target device/dtype.
        """
        return BatchSphericalImage(
            batch_value=to_torch(self.batch_value, device=device, dtype=dtype),
            vector=to_torch(self.vector, device=device, dtype=dtype),
            polar=to_torch(self.polar, device=device, dtype=dtype),
            mask=to_torch(self.mask, device=device, dtype=torch.bool),
        )

    def to_numpy(self) -> BatchSphericalImage:
        """
        Convert the data to numpy.ndarray.
        """
        return BatchSphericalImage(
            batch_value=to_numpy(self.batch_value),
            vector=to_numpy(self.vector),
            polar=to_numpy(self.polar),
            mask=to_numpy(self.mask),
        )


@multimethod
def concatenate(  # type: ignore
    images: tuple[SphericalImage] | list[SphericalImage],
) -> SphericalImage:
    """Concatenate channel dimension of input spherical images

    Args:
        images (tuple[SphericalImage] | list[SphericalImage]): tuple or list of spherical image

    Returns:
        SphericalImage: value channel concatenated
    """
    first_image = images[0]
    common_type = type(first_image.value)
    assert all(
        isinstance(image.value, common_type) for image in images[1:]
    ), "Cannot concatenate spherical images of different backend"

    assert_batch_allclose(
        [image.vector for image in images],
        "All spherical images must have consistent pixel location",
    )

    assert_batch_allclose(
        [image.mask for image in images],
        "All spherical images must have consistent mask",
    )

    return SphericalImage(
        value=concatenate([image.value for image in images], -1),
        vector=first_image.vector,
        polar=first_image.polar,
        mask=first_image.mask,
    )


@multimethod
def concatenate(
    batch_images: tuple[BatchSphericalImage] | list[BatchSphericalImage],
) -> BatchSphericalImage:
    """Concatenate channel dimension of input batch spherical images

    Args:
        batch_images (tuple[BatchSphericalImage] | list[BatchSphericalImage]): tuple or list of batch spherical image

    Returns:
        BatchSphericalImage: batch_value channel concatenated
    """
    first_image = batch_images[0]

    common_type = type(first_image.batch_value)
    assert all(
        isinstance(batch_image.batch_value, common_type)
        for batch_image in batch_images[1:]
    ), "Cannot concatenate batch spherical images of different backend"

    assert_batch_allclose(
        [image.vector for image in batch_images],
        "All batch spherical images must have consistent pixel location",
    )

    assert_batch_allclose(
        [image.mask for image in batch_images],
        "All batch spherical images must have consistent mask",
    )

    return BatchSphericalImage(
        batch_value=concatenate(
            [batch_image.batch_value for batch_image in batch_images], -1
        ),
        vector=first_image.vector,
        polar=first_image.polar,
        mask=first_image.mask,
    )


@multimethod
def split(  # type: ignore
    image: SphericalImage, indices_or_sections: int | list[int]
) -> list[SphericalImage]:
    """split the spherical image by value channel

    Args:
        image (SphericalImage): Input spherical image.
        indices_or_sections (int | list[int]): refer to numpy/torch's split

    Returns:
        list[SphericalImage]: list of splitted spherical images
    """

    return [
        SphericalImage(
            value=value,
            vector=image.vector,
            polar=image.polar,
            mask=image.mask,
        )
        for value in split(image.value, indices_or_sections, -1)
    ]


@multimethod
def split(
    batch_image: BatchSphericalImage, indices_or_sections: int | list[int]
) -> list[BatchSphericalImage]:
    """split the batch spherical image by batch_value channel

    Args:
        batch_image (BatchSphericalImage): Input batch spherical image.
        indices_or_sections (int | list[int]): refer to numpy/torch's split

    Returns:
        list[BatchSphericalImage]: list of splitted batch spherical images
    """

    return [
        BatchSphericalImage(
            batch_value=batch_value,
            vector=batch_image.vector,
            polar=batch_image.polar,
            mask=batch_image.mask,
        )
        for batch_value in split(batch_image.batch_value, indices_or_sections, -1)
    ]
