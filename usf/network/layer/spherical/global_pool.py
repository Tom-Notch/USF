#!/usr/bin/env python3
#
# Created on Sat Apr 19 2025 22:53:20
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
import torch
import torch.nn as nn

from usf.utils.spherical_image import BatchSphericalImage


class SphericalGlobalPool(nn.Module):
    """Apply global pooling to the batch_value"""

    def __init__(self, pool_type: str, *args, **kwargs):
        """
        Global Pooling Module

        Args:
            pool_type (str): Type of global pooling, must be one of ["max", "min", "mean"].
        """
        super().__init__()

        self.pool_type = pool_type
        if pool_type == "max":
            self.pool = torch.amax
        elif pool_type == "min":
            self.pool = torch.amin
        elif pool_type == "mean":
            self.pool = torch.mean
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Pool type name.
        """
        return f"type={self.pool_type}"

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Global pool ``batch_value`` along the spatial dim and broadcast back.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Output with spatially-constant pooled values.
        """
        return BatchSphericalImage(
            batch_value=self.pool(
                batch_spherical_image.batch_value, dim=-2, keepdim=True
            ).expand(-1, batch_spherical_image.num_pixels, -1),
            vector=batch_spherical_image.vector,
            mask=batch_spherical_image.mask,
        )
