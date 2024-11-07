#!/usr/bin/env python3
#
# Created on Fri Nov 22 2024 17:26:15
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import warnings
from typing import Any

import torch.nn as nn

from usf.sampler.location.location_sampler import LocationSampler
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage


class SphericalSampler(nn.Module):
    """Combined location + value sampler: resample a spherical image onto a new grid.

    Composes a ``LocationSampler`` (to generate target grid points) and a
    ``ValueSampler`` (to interpolate values onto that grid) into a single
    callable module.
    """

    def __init__(self, config: dict[str, Any], *args, **kwargs) -> None:
        """Initialize the combined spherical sampler.

        Args:
            config (dict[str, Any]): Configuration dict passed to both
                ``LocationSampler`` and ``ValueSampler``.
        """
        super().__init__()
        self.config = config

        if (
            self.config.get("reject_oo_fov_vector", True) is False
            and self.config.get("reject_oo_fov_value", False) is False
        ):
            warnings.warn(
                "not rejecting any out-of-fov vector or value, this will lead to thickened border",
                UserWarning,
                stacklevel=2,
            )

        self.location_sampler = LocationSampler(config)
        self.value_sampler = ValueSampler(config)
        self._init_args = args
        self._init_kwargs = kwargs

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        """Support pickling by returning the original constructor arguments.

        Returns:
            tuple[tuple, dict]: ``(positional_args, keyword_args)`` for ``__new__``.
        """
        # used by pickle to get args for __new__
        return ((self.config, *self._init_args), self._init_kwargs)

    def forward(
        self,
        batch_spherical_image: BatchSphericalImage | Any,
        *args,
        **kwargs,
    ) -> BatchSphericalImage | SphericalImage:
        """Sample values given spherical image

        Args:
            batch_spherical_image (BatchSphericalImage | Any): input spherical image to sample pixel values from

        Returns:
            SphericalImage: when input is a single spherical image
            BatchSphericalImage: otherwise
        """
        is_single_spherical_image = isinstance(batch_spherical_image, SphericalImage)

        # always process input as tensor BatchSphericalImage
        if not isinstance(batch_spherical_image, BatchSphericalImage):
            batch_spherical_image = BatchSphericalImage(batch_spherical_image)
        batch_spherical_image = batch_spherical_image

        output_vector = self.location_sampler(
            batch_spherical_image.vector,
            *args,
            **kwargs,
        )

        output_batch_spherical_image = self.value_sampler(
            batch_spherical_image,
            output_vector,
            *args,
            **kwargs,
        )

        return (
            output_batch_spherical_image[0]
            if is_single_spherical_image
            else output_batch_spherical_image
        )
