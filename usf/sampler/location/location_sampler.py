#!/usr/bin/env python3
#
# Created on Tue Nov 12 2024 17:39:55
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from cachetools import LFUCache
from matplotlib import cm
from multimethod import multimethod

from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.spherical import (
    average_pixel_area,
    cartesian2polar,
    compute_inside_fov_mask,
    normalize_cartesian,
    ripple_sort,
    unique_rows_preserve_order,
)
from usf.utils.torch_numpy import array_hash, copy_or_clone, to_numpy, to_torch

# Plasma color map to visualize index sequence of spherical sample locations
colormap = cm.get_cmap("plasma")


class LocationSampler(nn.Module, ABC):
    """Abstract base class and factory for sphere tessellation strategies.

    ``LocationSampler`` uses ``__new__`` as a factory: instantiation returns a
    concrete subclass selected by ``config["location_sampler"]`` (e.g.
    ``"icosahedron"``, ``"fibonacci"``, ``"healpix"``, etc.).

    **Subclass contract** — each tessellation backend must implement:

    - ``pixel_area2vec(average_pixel_area) -> torch.Tensor``:
      given a target average pixel area (in steradians), return a set of
      approximately-uniform unit Cartesian vectors on the sphere of shape
      ``(N, 3)`` whose density matches that area.

    Optionally override:

    - ``extra_args() -> str``: append backend-specific info to ``extra_repr``.

    The base class handles output caching, FoV masking, and dispatch between
    ``float`` (area), ``torch.Tensor``, and ``np.ndarray`` call signatures.
    """

    # each output vector is determined by input vector and resolution_factor
    # so cache the output vector to avoid redundant computation
    output_vector_cache = LFUCache(maxsize=500)

    _LOCATION_SAMPLER_TYPES = {
        "fibonacci": (".fibonacci", "Fibonacci"),
        "quasirandom": (".quasirandom", "QuasiRandom"),
        "healpix": (".healpix", "HEALPix"),
        "tetrahedron": (".tetrahedron", "Tetrahedron"),
        "hexahedron": (".hexahedron", "Hexahedron"),
        "octahedron": (".octahedron", "Octahedron"),
        "icosahedron": (".icosahedron", "Icosahedron"),
        "equirectangular": (".equirectangular", "Equirectangular"),
    }

    def __new__(cls, config: dict, *args, **kwargs) -> LocationSampler:
        """Factory constructor — resolve ``config["location_sampler"]`` to a concrete subclass.

        Args:
            config (dict): Must contain ``"location_sampler"`` (str) selecting
                the tessellation backend.

        Returns:
            LocationSampler: An instance of the selected concrete subclass.

        Raises:
            ValueError: If ``config`` is None, missing ``"location_sampler"``,
                or specifies an unsupported type.
        """
        if config is None:
            raise ValueError("config must be provided")

        if "location_sampler" not in config.keys():
            raise ValueError("location_sampler must be provided in config")

        location_sampler_type = config["location_sampler"]

        try:
            module_path, class_name = cls._LOCATION_SAMPLER_TYPES[location_sampler_type]
        except KeyError:
            raise ValueError(
                f"unsupported location sampler type {location_sampler_type}, supported types: {cls._LOCATION_SAMPLER_TYPES}"
            )

        # Import the module and get the class. This is done lazily
        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        return subclass_instance

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Store configuration and extract shared parameters.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.reject_oo_fov_vector = config.get("reject_oo_fov_vector", True)
        self.resolution_factor = config.get("resolution_factor", 1.0)
        self._init_args = args
        self._init_kwargs = kwargs

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        """Support pickling by returning the original constructor arguments.

        Returns:
            tuple[tuple, dict]: ``(positional_args, keyword_args)`` for ``__new__``.
        """
        # used by pickle to get args for __new__
        return ((self.config, *self._init_args), self._init_kwargs)

    def extra_repr(self) -> str:
        """Return compact string summary for ``print(module)``.

        Returns:
            str: Includes ``extra_args()`` output plus resolution/rejection flags.
        """
        extra_repr = []

        extra_args = self.extra_args()
        if extra_args:
            extra_repr.append(extra_args)

        extra_repr.append(
            f"resolution_factor={self.resolution_factor:.2f}"
            f", reject_oo_fov_vector={self.reject_oo_fov_vector}"
        )

        return ", ".join(extra_repr)

    def extra_args(self) -> str:
        """Optionally overridable function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        return ""

    @multimethod
    def forward(  # type: ignore
        self,
        output_average_pixel_area: float,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates locations on a sphere to sample pixel values from the input image, adaptively adjusts density (n_side) to match the resolution of the input image

        Args:
            output_average_pixel_area (float): desired output average area per vector

        Touches:
            self.output_vector_cache (dict): locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)

        Returns:
            torch.Tensor: locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)
        """
        input_vector_hash = (
            str(self.config["location_sampler"])
            + "_"
            + str("output_average_pixel_area")
            + "_"
            + str(output_average_pixel_area).replace(".", "dot")
        )
        buffer_name = f"output_vector_{input_vector_hash}"

        # Get cache and buffer (if they exist)
        buffer_entry = getattr(self, buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_vector_hash not in self.output_vector_cache
            and buffer_entry is not None
        ):
            output_vector = buffer_entry
            self.output_vector_cache[input_vector_hash] = output_vector
        elif input_vector_hash in self.output_vector_cache and buffer_entry is None:
            output_vector = self.output_vector_cache[input_vector_hash]

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        elif input_vector_hash not in self.output_vector_cache and buffer_entry is None:
            # Both are missing, so compute and cache the result
            # Get the sample locations from the average pixel area
            output_vector = to_torch(self.pixel_area2vec(output_average_pixel_area))
            # Normalize
            output_vector = normalize_cartesian(output_vector)
            # Remove duplicate points
            output_vector = unique_rows_preserve_order(output_vector, decimals=8)

            self.output_vector_cache[input_vector_hash] = output_vector

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        else:
            output_vector = buffer_entry

        return to_torch(output_vector)

    @multimethod
    def forward(  # noqa: F811 # type: ignore
        self,
        coordinates: torch.Tensor,
        resolution_factor: float | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates locations on a sphere to sample pixel values from the input image, adaptively adjusts density (n_side) to match the resolution of the input image

        Args:
            coordinates (torch.Tensor): spherical image's projected locations on a sphere, in polar coordinates (theta, phi), and in shape (..., 2)
            resolution_factor (float | None): factor to approximately adjust the resolution of the sample locations, default to 1.0
                                       if you want to half the number of sample locations, set this to 0.5

        Touches:
            self.output_vector_cache (dict): locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)

        Returns:
            torch.Tensor: locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)
        """
        resolution_factor = (
            self.resolution_factor if resolution_factor is None else resolution_factor
        )

        assert resolution_factor > 0.0, "resolution_factor must be greater than 0.0"

        # hash the input vector and resolution_factor to cache the output vector, sequence does not matter so ripple_sort
        input_vector_hash = (
            array_hash(ripple_sort(coordinates))
            + "_"
            + str(self.config["location_sampler"])
            + "_"
            + str(self.reject_oo_fov_vector)
            + "_"
            + (
                str(self.config["output_average_pixel_area"])
                if self.config.get("output_average_pixel_area", False)
                else str(resolution_factor)
            ).replace(".", "dot")
        )
        buffer_name = f"output_vector_{input_vector_hash}"

        # Get cache and buffer (if they exist)
        buffer_entry = getattr(self, buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_vector_hash not in self.output_vector_cache
            and buffer_entry is not None
        ):
            output_vector = buffer_entry

            self.output_vector_cache[input_vector_hash] = output_vector
        elif input_vector_hash in self.output_vector_cache and buffer_entry is None:
            output_vector = self.output_vector_cache[input_vector_hash]

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        elif input_vector_hash not in self.output_vector_cache and buffer_entry is None:
            # Both are missing, so compute and cache the result
            # determine if overridden
            if self.config.get("output_average_pixel_area", False):
                output_average_pixel_area = self.config["output_average_pixel_area"]
                assert (
                    0.0 < output_average_pixel_area < 4 * math.pi
                ), "output_average_pixel_area must be in (0, 4 * pi) when set"
            else:
                # Calculate the average pixel area and spherical polygons
                input_average_pixel_area = average_pixel_area(
                    to_numpy(copy_or_clone(coordinates))
                )  # ! scipy's spherical voronoi breaks when input is a large torch.Tensor
                # 4 pi is the unit sphere area
                output_average_pixel_area = min(
                    input_average_pixel_area / resolution_factor, 4 * math.pi
                )

            # Get the sample locations from the average pixel area
            output_vector = self(output_average_pixel_area).to(coordinates.device)

            # if panorama mode is not enabled, use nearest neighbor distance to reject out-of-view sample locations
            if self.reject_oo_fov_vector:
                inside_mask = compute_inside_fov_mask(output_vector, coordinates)
                output_vector = output_vector[inside_mask]

            self.output_vector_cache[input_vector_hash] = output_vector

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        else:
            output_vector = buffer_entry

        return to_torch(
            output_vector, device=coordinates.device, dtype=coordinates.dtype
        )

    @multimethod
    def forward(  # noqa: F811 # type: ignore
        self,
        coordinates: np.ndarray,
        resolution_factor: float | None = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Calculates locations on a sphere to sample pixel values from the input image, adaptively adjusts density (n_side) to match the resolution of the input image

        Args:
            coordinates (np.ndarray): spherical image's projected locations on a sphere, in polar coordinates (theta, phi), and in shape (..., 2)
            resolution_factor (float | None): factor to approximately adjust the resolution of the sample locations, default to 1.0
                                       if you want to half the number of sample locations, set this to 0.5

        Touches:
            self.sample_locations (np.ndarray): locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)

        Returns:
            np.ndarray: locations determined by the sampler, in cartesian coordinates (x, y, z), and in shape (N, 3)
        """
        resolution_factor = (
            self.resolution_factor if resolution_factor is None else resolution_factor
        )

        assert resolution_factor > 0.0, "resolution_factor must be greater than 0.0"

        # hash the input vector and resolution_factor to cache the output vector, sequence does not matter so ripple_sort
        input_vector_hash = (
            array_hash(ripple_sort(coordinates))
            + "_"
            + str(self.config["location_sampler"])
            + "_"
            + str(self.reject_oo_fov_vector)
            + "_"
            + (
                str(self.config["output_average_pixel_area"])
                if self.config.get("output_average_pixel_area", False)
                else str(resolution_factor)
            ).replace(".", "dot")
        )
        buffer_name = f"output_vector_{input_vector_hash}"

        # Get cache and buffer (if they exist)
        buffer_entry = getattr(self, buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_vector_hash not in self.output_vector_cache
            and buffer_entry is not None
        ):
            output_vector = buffer_entry

            self.output_vector_cache[input_vector_hash] = output_vector
        elif input_vector_hash in self.output_vector_cache and buffer_entry is None:
            output_vector = self.output_vector_cache[input_vector_hash]

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        elif input_vector_hash not in self.output_vector_cache and buffer_entry is None:
            # Both are missing, so compute and cache the result
            # determine if overridden
            if self.config.get("output_average_pixel_area", False):
                output_average_pixel_area = self.config["output_average_pixel_area"]
                assert (
                    0.0 < output_average_pixel_area < 4 * math.pi
                ), "output_average_pixel_area must be in (0, 4 * pi) when set"
            else:
                # Calculate the average pixel area and spherical polygons
                input_average_pixel_area = average_pixel_area(coordinates)
                # 4 pi is the unit sphere area
                output_average_pixel_area = min(
                    input_average_pixel_area / resolution_factor, 4 * math.pi
                )

            # Get the sample locations from the average pixel area
            output_vector = self(output_average_pixel_area)

            # if reject_oo_fov_vector is enabled, use nearest neighbor distance to reject out-of-view sample locations
            if self.reject_oo_fov_vector:
                inside_mask = compute_inside_fov_mask(
                    output_vector, to_torch(coordinates)
                )
                output_vector = output_vector[inside_mask]

            self.output_vector_cache[input_vector_hash] = output_vector

            if is_cache_enabled():
                self.register_buffer(buffer_name, output_vector)
        else:
            output_vector = buffer_entry

        return to_numpy(output_vector)

    @abstractmethod
    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Gets co-centric unit 3D vectors for the sample locations on a sphere given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Raises:
            NotImplementedError: This method must be implemented in the derived class

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )

    def pixel_area2polar(self, average_pixel_area: float) -> torch.Tensor:
        """Gets polar coordinates for the sample locations on a sphere given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 2) containing co-centric polar coordinates (theta, phi)
        """
        vector = self.pixel_area2vec(average_pixel_area)
        return cartesian2polar(vector)


register_cache(
    "LocationSampler.output_vector_cache", LocationSampler.output_vector_cache
)
