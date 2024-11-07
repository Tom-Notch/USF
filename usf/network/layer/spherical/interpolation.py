#!/usr/bin/env python3
#
# Created on Fri Oct 31 2025 16:21:48
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.spherical_image import BatchSphericalImage


class SphericalInterpolation(nn.Module):
    """Spherical interpolation layer that resamples a ``BatchSphericalImage`` onto a new grid.

    Wraps a ``ValueSampler`` to interpolate pixel values from the input point
    cloud onto a fixed ``output_vector`` grid. Useful for changing resolution
    or projecting between different sphere tessellations.
    """

    def __init__(
        self, config: dict | None = None, output_vector: torch.Tensor | None = None
    ):
        """Modularized interpolation layer

        Args:
            config (dict | None, optional): config for value sampler. Defaults to None.
            output_vector (torch.Tensor | None, optional): output vector. Defaults to None.
        """
        super().__init__()
        self.sampler = ValueSampler(
            config=(
                config
                if isinstance(config, dict)
                else {"value_sampler": "radial_basis_function"}
            )
        )
        self.set_output_vector(output_vector)

    def set_output_vector(
        self, output_vector: torch.Tensor | None
    ) -> SphericalInterpolation:
        """Set output vector

        Args:
            output_vector (torch.Tensor | None): in shape (N, 3), each row is a cartesian coordinate

        Touches:
            self.output_vector: internal record

        Returns:
            self
        """
        if output_vector is not None:
            assert (
                output_vector.shape[-1] == 3
            ), f"Output_vector has last dimension = {output_vector.shape[-1]} which is not Cartesian"

            self.register_buffer(
                "output_vector",
                output_vector.detach().clone(),
            )

        elif getattr(self, "output_vector", None) is not None:
            del self.output_vector

        return self

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Interpolate values onto the configured output grid.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Resampled output, or passthrough if ``output_vector`` is not set.
        """
        output_vector = getattr(self, "output_vector", None)
        if output_vector is None:
            warnings.warn(
                "output vector not set, returning input batch_spherical_image",
                UserWarning,
                stacklevel=2,
            )
            return batch_spherical_image
        else:
            return self.sampler(batch_spherical_image, output_vector)
