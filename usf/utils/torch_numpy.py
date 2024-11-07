#!/usr/bin/env python3
#
# Created on Thu Feb 06 2025 11:51:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import hashlib
import random
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from multimethod import multimethod
from omegaconf import ListConfig


def autopad(
    kernel_size: int | list[int],
    padding: int | None = None,
    dilation: int = 1,
) -> int | list[int]:
    """Automatically compute padding to maintain 'same' output shape in convolutions.

    Args:
        kernel_size (int | list[int]): Convolution kernel size(s).
        padding (int | None, optional): Explicit padding override. If provided, returned as-is.
            Defaults to None ('same' auto-compute).
        dilation (int, optional): Convolution dilation factor. Defaults to 1.

    Returns:
        int | list[int]: Padding value(s) that preserve spatial dimensions.
    """
    if dilation > 1:
        kernel_size = (
            dilation * (kernel_size - 1) + 1
            if isinstance(kernel_size, int)
            else [dilation * (size - 1) + 1 for size in kernel_size]
        )  # Adjust kernel size for dilation

    return (
        padding
        if padding is not None
        else (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else [size // 2 for size in kernel_size]
        )
    )


@contextmanager
def fixed_seed(
    seed: int | None = None,
    *,
    numpy: bool = True,
    python: bool = True,
    device_scope: str = "auto",
    cuda_devices: Iterable[int] | None = None,
):
    """
    Temporarily set a deterministic RNG seed and restore all states afterward.

    Args:
        seed (int | None, optional): If None, this is a no-op. If int, applies deterministic seeding. Defaults to None.
        numpy (bool, optional): Snapshot/seed/restore NumPy RNG. Defaults to True.
        python (bool, optional): Snapshot/seed/restore Python's `random` RNG. Defaults to True.
        device_scope (str, optional): Seeding scope for CUDA devices. Defaults to "auto".
            - "auto": CPU only in DataLoader workers; CPU+CUDA in main process if CUDA is initialized.
            - "cpu":  CPU only everywhere (never touches CUDA).
            - "cuda": CUDA only (plus CPU); requires CUDA to be initialized; avoid in workers.
            - "both": CPU+CUDA unconditionally; avoid in workers; will seed CUDA only if initialized.
        cuda_devices (Iterable[int] | None, optional): Explicit iterable of CUDA device indices to include.
            If None, uses all initialized devices when CUDA is actually initialized and device_scope allows it.
            Defaults to None.
    """
    if seed is None:
        yield
        return

    # Detect if we are in a DataLoader worker
    try:
        in_worker = torch.utils.data.get_worker_info() is not None
    except Exception:
        in_worker = False

    # Decide whether to include CUDA RNG
    if device_scope == "cpu":
        touch_cuda = False
    elif device_scope in ("cuda", "both"):
        touch_cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    else:  # "auto"
        touch_cuda = (
            (not in_worker)
            and torch.cuda.is_available()
            and torch.cuda.is_initialized()
        )

    devices: list[int] = []
    if touch_cuda:
        if cuda_devices is not None:
            devices = list(cuda_devices)
        else:
            try:
                devices = list(range(torch.cuda.device_count()))
            except Exception:
                devices = []

    # Snapshot NumPy / Python RNG states if requested
    np_state = None
    py_state = None
    if numpy:
        np_state = np.random.get_state()
    if python:
        py_state = random.getstate()

    # Torch RNG snapshot/restore
    with torch.random.fork_rng(devices=devices, enabled=True):
        # Apply seeds
        torch.manual_seed(seed)
        if touch_cuda:
            torch.cuda.manual_seed_all(seed)
        if numpy:
            import numpy as _np

            _np.random.seed(seed)
        if python:
            import random as _random

            _random.seed(seed)

        try:
            yield

        finally:
            # fork_rng auto-restores torch states, we restore NumPy/Python manually
            if numpy and np_state is not None:
                import numpy as _np

                _np.random.set_state(np_state)
            if python and py_state is not None:
                import random as _random

                _random.setstate(py_state)


def string_to_seed(s: str, max_value: int = 2**32 - 1) -> int:
    """Convert a string to a deterministic integer seed within [0, max_value)."""
    hash_digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(hash_digest, 16) % max_value


def million_trainable_params(model: nn.Module) -> float:
    """Count trainable parameters in millions.

    Args:
        model (nn.Module): PyTorch module.

    Returns:
        float: Number of trainable parameters divided by 1e6.
    """
    total_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total_params / 1e6


def count_layers(model: nn.Module, layer_type: type) -> int:
    """Count the number of sub-modules of a given type.

    Args:
        model (nn.Module): PyTorch module to inspect (recursive).
        layer_type (type): Module class to count (e.g. ``nn.Linear``).

    Returns:
        int: Number of matching sub-modules.
    """
    return sum(1 for layer in model.modules() if isinstance(layer, layer_type))


@multimethod
def assert_batch_allclose(  # type: ignore
    batch: Iterable[torch.Tensor],
    error_msg: str,
    *args,
    **kwargs,
) -> None:
    """Assert all tensors in the list are close to the first one.
    Uses torch._assert with 0-D bool tensor => safe under torch.compile/make_fx.

    Args:
        batch (Iterable[torch.Tensor]): list of tensors
        error_msg (str): error message.
    """
    tensors = list(batch)
    if len(tensors) <= 1:
        return

    first_tensor = tensors[0]
    _device, _dtype = first_tensor.device, first_tensor.dtype
    condition = torch.ones([], dtype=torch.bool, device=_device)

    for tensor in tensors[1:]:
        same_shape = first_tensor.shape == tensor.shape
        if not same_shape:
            condition = torch.zeros([], dtype=torch.bool, device=_device)
            break
        close = first_tensor.isclose(
            tensor.to(dtype=_dtype, device=_device), *args, **kwargs
        ).all()
        condition = condition & close

    torch._assert(condition, error_msg)


@multimethod
def assert_batch_allclose(
    batch: Iterable[np.ndarray],
    error_msg: str,
    *args,
    **kwargs,
) -> None:
    """Assert all numpy arrays in the list are close to the first one

    Args:
        batch (Iterable[np.ndarray]): list of numpy arrays
        error_msg (str): error message
    """
    arrays = list(batch)
    first_array = arrays[0]
    assert all(
        np.allclose(array, first_array, *args, **kwargs) for array in arrays[1:]
    ), error_msg


@multimethod
def allclose(a: torch.Tensor, b: torch.Tensor, *args, **kwargs) -> bool:  # type: ignore
    """Check if two tensors are element-wise close (torch dispatch).

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        bool: True if all elements satisfy the closeness criterion.
    """
    return torch.allclose(a, b, *args, **kwargs)


@multimethod
def allclose(a: np.ndarray, b: np.ndarray, *args, **kwargs) -> bool:
    """Check if two arrays are element-wise close (numpy dispatch).

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        bool: True if all elements satisfy the closeness criterion.
    """
    return np.allclose(a, b, *args, **kwargs)


@multimethod
def detect_nan(x: torch.Tensor, variable_name: str, print_indices: bool = False) -> bool:  # type: ignore
    """Detect NaN values in a tensor and optionally print their indices (torch dispatch).

    Args:
        x (torch.Tensor): Tensor to check.
        variable_name (str): Label printed in diagnostic messages.
        print_indices (bool, optional): If True, print the indices of NaN elements.
            Defaults to False.

    Returns:
        bool: True if any NaN is found.
    """
    if torch.isnan(x).any():
        if print_indices:
            nan_indices = torch.nonzero(torch.isnan(x))
            print(f"{variable_name} contains nan at indices:\n{nan_indices.tolist()}")
        else:
            print(f"{variable_name} contains nan")
        return True
    else:
        return False


@multimethod
def detect_nan(x: np.ndarray, variable_name: str, print_indices: bool = False) -> bool:
    """Detect NaN values in an array and optionally print their indices (numpy dispatch).

    Args:
        x (np.ndarray): Array to check.
        variable_name (str): Label printed in diagnostic messages.
        print_indices (bool, optional): If True, print the indices of NaN elements.
            Defaults to False.

    Returns:
        bool: True if any NaN is found.
    """
    if np.isnan(x).any():
        if print_indices:
            nan_indices = np.argwhere(np.isnan(x))
            print(f"{variable_name} contains nan at indices:\n{nan_indices.tolist()}")
        else:
            print(f"{variable_name} contains nan")
        return True
    else:
        return False


def module_weights_hash(module: torch.nn.Module) -> str:
    """
    Computes an MD5 hash for the weights of a module.

    Args:
        module (torch.nn.Module): The PyTorch module whose weights are to be hashed.

    Returns:
        str: An MD5 hash representing the weights of the module.
    """
    md5 = hashlib.md5()
    for _, weight in module.named_parameters():
        # Move to CPU and cast to a fixed dtype for consistency.
        array = (
            to_numpy(copy_or_clone(weight)).astype(np.float32).astype("<f4", copy=False)
        )
        md5.update(array.tobytes())
    return md5.hexdigest()


@multimethod
def array_hash(array: np.ndarray) -> str:  # type: ignore
    """Hash the numpy array based on its content, cross-comparable with torch tensors

    Args:
        array (np.ndarray): Input array.

    Returns:
        str: md5
    """
    assert isinstance(array, np.ndarray), "Expect input to be a numpy array"

    # Standardize: copy the array, cast to float32, and force little-endian byte order.
    standardized = np.array(array, copy=True)
    standardized = standardized.astype(np.float32)  # Convert to float32
    standardized = standardized.astype("<f4", copy=False)  # Force little-endian

    return hashlib.md5(standardized.tobytes()).hexdigest()


@multimethod
def array_hash(array: torch.Tensor) -> str:
    """Hash the tensor based on its content, cross-comparable with numpy arrays

    Args:
        array (torch.Tensor): Input tensor.

    Returns:
        str: md5
    """
    assert isinstance(array, torch.Tensor), "Expect input to be a torch.Tensor"

    return array_hash(to_numpy(copy_or_clone(array)))


@multimethod
def array_size_MiB(array: torch.Tensor) -> float:  # type: ignore
    """
    Compute the memory size of a tensor in MiB, handling both dense and sparse tensors.

    Args:
        array (torch.Tensor): Input tensor (dense or sparse).

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, torch.Tensor), "Expect input to be a torch.Tensor"

    if array.is_sparse:
        # Sparse tensor: compute size using indices and values
        array = array.coalesce()
        indices = array.indices()
        values = array.values()
        total_bytes = (indices.numel() * indices.element_size()) + (
            values.numel() * values.element_size()
        )
    else:
        # Dense tensor: compute size based on total elements and element size
        total_bytes = array.numel() * array.element_size()

    return total_bytes / (2**20)  # Convert to MiB


@multimethod
def array_size_MiB(array: np.ndarray) -> float:  # type: ignore
    """
    Compute the memory size of an array in MiB.

    Supports both dense numpy arrays and scipy sparse matrices.

    Args:
        array (np.ndarray): Input array.

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, np.ndarray), "Expect input to be a numpy array"

    total_bytes = array.nbytes

    return total_bytes / (2**20)


@multimethod
def array_size_MiB(array: sp.spmatrix) -> float:
    """
    Compute the memory size of a sparse matrix in MiB.

    Args:
        array (sp.spmatrix): Input sparse matrix.

    Returns:
        float: Memory size in MiB.
    """
    assert isinstance(array, sp.spmatrix), "Expect input to be a scipy sparse matrix"

    total_bytes = array.data.nbytes + array.indices.nbytes + array.indptr.nbytes

    return total_bytes / (2**20)


@multimethod
def to_torch(  # type: ignore
    array: ListConfig,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert Hydra ListConfig to torch tensor.

    Args:
        array (ListConfig): Raw list read from Hydra yaml.
        device (torch.device | None, optional): Target device. Defaults to None.
        dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

    Returns:
        torch.Tensor: Converted tensor.
    """
    return torch.Tensor(list(array)).to(device=device, dtype=dtype)


@multimethod
def to_torch(  # type: ignore
    array: list[int | float | bool],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert Python list to torch tensor.

    Args:
        array (list[int | float | bool]): Python list of numeric or boolean values.
        device (torch.device | None, optional): Target device. Defaults to None.
        dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

    Returns:
        torch.Tensor: Converted tensor.
    """
    return torch.Tensor(array).to(device=device, dtype=dtype)


@multimethod
def to_torch(  # type: ignore
    array: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert numpy array to torch tensor.

    Args:
        array (np.ndarray): Input numpy array.
        device (torch.device | None, optional): Target device. Defaults to None.
        dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

    Returns:
        torch.Tensor: Converted tensor (zero-copy when possible via ``torch.from_numpy``).
    """

    return torch.from_numpy(array).to(device=device, dtype=dtype)


@multimethod
def to_torch(
    array: torch.Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Move/cast an existing torch tensor to the specified device and dtype.

    Args:
        array (torch.Tensor): Input tensor.
        device (torch.device | None, optional): Target device. Defaults to None.
        dtype (torch.dtype | None, optional): Target dtype. Defaults to None.

    Returns:
        torch.Tensor: The tensor on the target device/dtype.
    """

    return array.to(device=device, dtype=dtype)


@multimethod
def to_numpy(array: ListConfig) -> np.ndarray:  # type: ignore
    """Convert Hydra List to numpy array

    Args:
        array (ListConfig): raw list read from Hydra yaml

    Returns:
        np.ndarray
    """
    return np.array(list(array))


@multimethod
def to_numpy(array: list[int | float | bool]) -> np.ndarray:  # type: ignore
    """Convert python array to numpy array

    Args:
        array (list[int | float | bool]): python array of int, float, or bool

    Returns:
        np.ndarray
    """
    return np.array(array)


@multimethod
def to_numpy(array: np.ndarray) -> np.ndarray:  # type: ignore
    """Convert numpy array to numpy array

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray
    """

    return array


@multimethod
def to_numpy(array: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array

    Args:
        array (torch.Tensor): Input tensor.

    Returns:
        np.ndarray
    """
    if array.dtype == torch.bfloat16:
        # Convert bfloat16 to float32 before converting to numpy
        array = array.to(torch.float32)

    return array.numpy(force=True)


@multimethod
def copy_or_clone(array: np.ndarray) -> np.ndarray:  # type: ignore
    """Copy numpy array

    Args:
        array (np.ndarray): Input array.

    Returns:
        np.ndarray
    """

    return array.copy()


@multimethod
def copy_or_clone(array: torch.Tensor) -> torch.Tensor:
    """Clone torch tensor

    Args:
        array (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor
    """

    return array.clone()


@multimethod
def stack(arrays: tuple[np.ndarray] | list[np.ndarray], dim: int) -> np.ndarray:  # type: ignore
    """Stack numpy array

    Args:
        arrays (tuple[np.ndarray] | list[np.ndarray]): Input arrays.
        dim (int): dimension, translated into axis

    Returns:
        np.ndarray
    """
    return np.stack(arrays, axis=dim)


@multimethod
def stack(arrays: tuple[torch.Tensor] | list[torch.Tensor], dim: int) -> torch.Tensor:
    """Stack torch tensor

    Args:
        arrays (tuple[torch.Tensor] | list[torch.Tensor]): Input tensors.
        dim (int): dimension

    Returns:
        torch.Tensor
    """
    return torch.stack(arrays, dim=dim)


@multimethod
def concatenate(arrays: tuple[np.ndarray] | list[np.ndarray], dim: int) -> np.ndarray:  # type: ignore
    """Concatenate numpy array

    Args:
        arrays (tuple[np.ndarray] | list[np.ndarray]): Input arrays.
        dim (int): dimension, translated into axis

    Returns:
        np.ndarray
    """
    return np.concatenate(arrays, axis=dim)


@multimethod
def concatenate(
    arrays: tuple[torch.Tensor] | list[torch.Tensor], dim: int
) -> torch.Tensor:
    """Concatenate torch tensor

    Args:
        arrays (tuple[torch.Tensor] | list[torch.Tensor]): Input tensors.
        dim (int): dimension

    Returns:
        torch.Tensor
    """
    return torch.cat(arrays, dim=dim)


@multimethod
def split(  # type: ignore
    array: np.ndarray, indices_or_sections: int | list[int], dim: int
) -> list[np.ndarray]:
    """
    Split a numpy array along a specified dimension.

    Args:
        array (np.ndarray): The input array.
        indices_or_sections (int | list[int]): Either an integer indicating the number
            of equal splits or a list of indices where the split should occur.
        dim (int): The dimension along which to split (mapped to np.split's axis).

    Returns:
        list[np.ndarray]: A list of numpy arrays after splitting.
    """
    return np.split(array, indices_or_sections, axis=dim)


@multimethod
def split(
    array: torch.Tensor, indices_or_sections: int | list[int], dim: int
) -> list[torch.Tensor]:
    """
    Split a torch tensor along a specified dimension.

    Args:
        array (torch.Tensor): The input tensor.
        indices_or_sections (int | list[int]): Either an integer for uniform splits or
            a list of sizes for each split.
        dim (int): The dimension along which to split.

    Returns:
        list[torch.Tensor]: A list of torch tensors after splitting.
    """
    return list(torch.split(array, indices_or_sections, dim=dim))
