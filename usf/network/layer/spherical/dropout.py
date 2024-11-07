#!/usr/bin/env python3
#
# Created on Thu Oct 02 2025 18:48:15
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch.nn as nn
import torch.nn.functional as F

from usf.utils.spherical_image import BatchSphericalImage


class SphericalDropout(nn.Module):
    """Apply dropout with Bernoulli probability on all elements of batch_value regardless of batch, spatial, or channel dimension"""

    def __init__(self, p: float, inplace: bool = False):
        """
        Spherical Dropout Module

        Args:
            p (float): dropout rate
            inplace (bool): whether to do it inplace. Default to False.
        """
        super().__init__()

        assert 0.0 < p < 1.0, f"p must be in (0.0, 1.0), got p == {p}"

        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Dropout probability and inplace flag.
        """
        return f"p={self.p}, inplace={self.inplace}"

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Apply dropout to ``batch_value``.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Cloned output with dropout applied to ``batch_value``.
        """
        output_batch_spherical_image = batch_spherical_image.clone()
        output_batch_spherical_image.batch_value = F.dropout(
            output_batch_spherical_image.batch_value,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )
        return output_batch_spherical_image
