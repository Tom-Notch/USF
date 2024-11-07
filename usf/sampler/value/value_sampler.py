#!/usr/bin/env python3
#
# Created on Fri Nov 22 2024 16:43:58
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from multimethod import multimethod

from usf.utils.spherical import compute_inside_fov_mask
from usf.utils.spherical_image import BatchSphericalImage


class ValueSampler(nn.Module, ABC):
    """Abstract base class and factory for spherical value interpolation methods.

    ``ValueSampler`` uses ``__new__`` as a factory: instantiation returns a
    concrete subclass selected by ``config["value_sampler"]`` (e.g.
    ``"nearest_neighbor"``, ``"radial_basis_function"``).

    **Subclass contract** — each interpolation backend must implement:

    - ``_sample(batch_spherical_image, output_vectors) -> BatchSphericalImage``:
      interpolate ``batch_spherical_image.batch_value`` from its current point
      cloud onto the ``output_vectors`` grid. Must handle batched inputs.

    Optionally override:

    - ``extra_args() -> str``: append backend-specific info to ``extra_repr``.

    The base class handles FoV masking (``reject_oo_fov_value``) and dispatch
    between ``torch.Tensor`` and ``np.ndarray`` call signatures.
    """

    _VALUE_SAMPLER_TYPES = {
        "nearest_neighbor": (".nearest_neighbor", "NearestNeighbor"),
        "radial_basis_function": (".radial_basis_function", "RadialBasisFunction"),
    }

    def __new__(cls, config: dict, *args, **kwargs) -> ValueSampler:
        """Factory constructor — resolve ``config["value_sampler"]`` to a concrete subclass.

        Args:
            config (dict): Must contain ``"value_sampler"`` (str) selecting the
                interpolation backend.

        Returns:
            ValueSampler: An instance of the selected concrete subclass.

        Raises:
            ValueError: If ``config`` is None, missing ``"value_sampler"``,
                or specifies an unsupported type.
        """
        if config is None:
            raise ValueError("config must be provided")

        if "value_sampler" not in config.keys():
            raise ValueError("value_sampler must be provided in config")

        type = config["value_sampler"]

        try:
            module_path, class_name = cls._VALUE_SAMPLER_TYPES[type]
        except KeyError:
            raise ValueError(
                f"unsupported value sampler type {type}, supported types: {cls._VALUE_SAMPLER_TYPES}"
            )

        # Import the module and get the class. This is done lazily
        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        return subclass_instance

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Store configuration and extract shared parameters.

        Args:
            config (dict): Value sampler configuration dictionary.
        """
        super().__init__()
        self.config = config
        self.reject_oo_fov_value = self.config.get("reject_oo_fov_value", True)
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
            str: Includes ``extra_args()`` output plus rejection flag.
        """
        extra_repr = []

        extra_args = self.extra_args()
        if extra_args:
            extra_repr.append(extra_args)

        extra_repr.append(f"reject_oo_fov_value={self.reject_oo_fov_value}")

        return ", ".join(extra_repr)

    def extra_args(self) -> str:
        """Optionally overridable function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        return ""

    @multimethod
    def forward(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: np.ndarray,
        *args,
        **kwargs,
    ) -> BatchSphericalImage:
        """sample pixel values with specified sampling method given the input image

        Args:
            batch_spherical_image (BatchSphericalImage): input spherical image to sample pixel values
            sample_vector (np.ndarray): numpy array of shape (..., 3) containing cartesian coordinates (x, y, z) to sample to

        Returns:
            BatchSphericalImage
        """
        sample_vector = sample_vector.reshape(-1, 3)

        # sample value
        batch_value = self.sample_value(
            batch_spherical_image,
            sample_vector,
            *args,
            **kwargs,
        )

        if self.reject_oo_fov_value:
            inside_mask = compute_inside_fov_mask(
                sample_vector, batch_spherical_image.apply_mask().vector
            )
        else:
            inside_mask = np.ones(sample_vector.shape[0], dtype=bool)

        return BatchSphericalImage(
            batch_value=batch_value,
            vector=sample_vector,
            mask=inside_mask,
        )

    @multimethod
    # @torch.compiler.disable(recursive=False)
    def forward(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: torch.Tensor,
        *args,
        **kwargs,
    ) -> BatchSphericalImage:
        """sample pixel values with specified sampling method given the input image

        Args:
            batch_spherical_image (BatchSphericalImage): input spherical image to sample pixel values
            sample_vector (torch.Tensor): torch.Tensor of shape (..., 3) containing cartesian coordinates (x, y, z) to sample to

        Returns:
            BatchSphericalImage
        """
        sample_vector = sample_vector.view(-1, 3).to(
            device=batch_spherical_image.vector.device
        )

        # sample value
        batch_value = self.sample_value(
            batch_spherical_image,
            sample_vector,
            *args,
            **kwargs,
        )

        if self.reject_oo_fov_value:
            inside_mask = compute_inside_fov_mask(
                sample_vector, batch_spherical_image.apply_mask().vector
            )
        else:
            inside_mask = torch.ones(
                sample_vector.shape[0],
                dtype=torch.bool,
                device=sample_vector.device,
            )

        return BatchSphericalImage(
            batch_value=batch_value,
            vector=sample_vector,
            mask=inside_mask,
        )

    @abstractmethod
    @multimethod
    @torch.no_grad()
    def sample_value(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """sample pixel values with specified sampling method given the input image

        Args:
            batch_spherical_image (BatchSphericalImage): masked input spherical image to sample pixel values from
            sample_vector (np.ndarray): numpy array of shape (..., 3) containing cartesian coordinates (x, y, z) to sample to

        Raises:
            NotImplementedError: This method must be implemented in the derived class

        Returns:
            np.ndarray: output value matching shape of sample_vector
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )

    @abstractmethod
    @multimethod
    def sample_value(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """sample pixel values with specified sampling method given the input image

        Args:
            batch_spherical_image (BatchSphericalImage): masked input spherical image to sample pixel values from
            sample_vector (torch.Tensor): torch.Tensor of shape (..., 3) containing cartesian coordinates (x, y, z) to sample to

        Raises:
            NotImplementedError: This method must be implemented in the derived class

        Returns:
            torch.Tensor: output value matching shape of sample_vector
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )
