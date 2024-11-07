#!/usr/bin/env python3
#
# Created on Fri Aug 29 2025 14:44:11
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn

from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage


class GenericSphericalConv(nn.Module, ABC):
    """Abstract base class and factory for spherical convolution backends.

    ``GenericSphericalConv`` uses ``__new__`` as a factory: instantiation always
    returns a concrete backend subclass selected by the ``backend`` keyword
    argument (one of ``"circle"``, ``"ripple"``, ``"spherical"``, ``"wave"``).

    **Subclass contract** — each backend must implement:

    - ``__init__``: accept ``in_channels``, ``out_channels``, ``radius``, and
      backend-specific kwargs; call ``super().__init__()`` which stores
      ``_init_args`` / ``_init_kwargs`` for pickling.
    - ``forward(batch_spherical_image) -> BatchSphericalImage``: the main
      convolution pass — receives an input batch of spherical images and
      produces an output batch with (potentially different) point locations
      and channel counts.
    - ``report_metrics(**kwargs) -> dict[str, Any] | None``: return diagnostic
      statistics about the most recent collection matrix (e.g. sparsity,
      cache hit rate). May return ``None`` if no statistics are available.
    - ``visualize_kernel(**kwargs) -> SphericalImage``: render the learned
      kernel weights onto a sphere for visualization / debugging.

    Backends may also override ``set_output_vector`` if they need to
    invalidate cached collection matrices when the output grid changes.
    """

    _BACKENDS = {
        "circle": ("._old.circle_cnn", "CircleConv"),
        "ripple": (".backend.ripple_cnn", "RippleConv"),
        "spherical": (".backend.spherical_cnn", "SphericalConv"),
        "wave": (".backend.wave_cnn", "WaveConv"),
    }

    def __new__(
        cls, backend: str | None = None, *args, **kwargs
    ) -> GenericSphericalConv:
        """Factory constructor — resolve ``backend`` to a concrete subclass and return it.

        Args:
            backend (str | None): Backend identifier (``"circle"``, ``"ripple"``,
                ``"spherical"``, or ``"wave"``). Required.

        Returns:
            GenericSphericalConv: An instance of the concrete backend subclass.
        """
        assert (
            backend is not None
        ), "GenericSphericalConv requires keyword argument 'backend'"
        assert (
            backend in cls._BACKENDS
        ), f"Unsupported backend {backend}. Currently supports {cls._BACKENDS.keys()}"

        module_path, class_name = cls._BACKENDS[backend]
        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        object.__setattr__(subclass_instance, "backend", backend)
        return subclass_instance

    def __init__(self, *args, **kwargs) -> None:
        """Store constructor positional/keyword args for pickle support."""
        super().__init__()
        self._init_args = args
        self._init_kwargs = kwargs

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        """Support pickling by returning the original constructor arguments.

        Returns:
            tuple[tuple, dict]: ``(positional_args, keyword_args)`` for ``__new__``.
        """
        return (self.backend, *self._init_args), self._init_kwargs

    def _reference_tensor(self, *hints: torch.Tensor | None) -> torch.Tensor:
        """Find a reference tensor to infer dtype and device.

        Checks, in order: explicit hints, module parameters, module buffers,
        then falls back to an empty CPU tensor.

        Returns:
            torch.Tensor: A tensor on the target device/dtype.
        """
        # 1) prefer any explicit hints (e.g., an input tensor)
        for t in hints:
            if isinstance(t, torch.Tensor):
                return t
        # 2) any parameter (recurse=True reaches deep children)
        parameter = next(self.parameters(recurse=True), None)
        if parameter is not None:
            return parameter
        # 3) any buffer
        buffer = next(self.buffers(recurse=True), None)
        if buffer is not None:
            return buffer
        # 4) last resort: empty CPU tensor
        return torch.empty(0)

    @property
    def device(self) -> torch.device:
        """Inferred device of this module (from parameters, buffers, or CPU fallback).

        Returns:
            torch.device: The device.
        """
        return self._reference_tensor().device

    @property
    def dtype(self) -> torch.dtype:
        """Inferred dtype of this module (from parameters, buffers, or default).

        Returns:
            torch.dtype: The dtype.
        """
        return self._reference_tensor().dtype

    def set_output_vector(
        self, output_vector: torch.Tensor | None
    ) -> GenericSphericalConv:
        """Set a fixed output vector that overrides any existing rule

        Args:
            output_vector (torch.Tensor | None): in shape (N, 3), each row is a cartesian coordinate

        Touches:
            self.override_output_vector: internal record

        Returns:
            self
        """
        if output_vector is not None:
            assert (
                output_vector.shape[-1] == 3
            ), f"Output_vector has last dimension = {output_vector.shape[-1]} which is not Cartesian"

            self.register_buffer(
                "override_output_vector",
                output_vector.detach().clone(),
            )

            if getattr(self, "location_sampler", None) is not None:
                del self.location_sampler
        elif getattr(self, "override_output_vector", None) is not None:
            del self.override_output_vector

            if getattr(self, "location_sampler", None) is None:
                self.location_sampler = LocationSampler(
                    config={
                        "location_sampler": getattr(
                            self, "location_sampler_type", "icosahedron"
                        ),
                        "resolution_factor": getattr(self, "resolution_factor", 1.0),
                        "reject_oo_fov_vector": getattr(
                            self, "reject_oo_fov_vector", True
                        ),
                    }
                )

        return self

    @abstractmethod
    def forward(
        self,
        batch_spherical_image: BatchSphericalImage,
    ) -> BatchSphericalImage:
        """Forward pass of GenericSphericalConv

        Args:
            batch_spherical_image (BatchSphericalImage): Input

        Returns:
            BatchSphericalImage: Output
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )

    @abstractmethod
    def report_metrics(self, *args, **kwargs) -> dict[str, Any] | None:
        """Report statistics on the latest collection matrix for GenericSphericalConv

        Returns:
            dict[str, Any] | None: optional metrics
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )

    @abstractmethod
    def visualize_kernel(self, *args, **kwargs) -> SphericalImage:
        """Visualize the kernel on a sphere

        Returns:
            SphericalImage: A visualizable spherical image containing the kernel
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )
