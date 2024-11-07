#!/usr/bin/env python3
#
# Created on Tue Feb 18 2025 12:41:27
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch.nn as nn

from usf.utils.spherical_image import BatchSphericalImage


class SphericalConv1x1(nn.Module):
    """Per-pixel 1x1 convolution on spherical signals.

    Applies a learned linear transform to each pixel independently via
    ``Conv1d(kernel_size=1)`` over the channel dimension. Geometry fields
    (vector, polar, mask) pass through unchanged.
    """

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True, *args, **kwargs
    ):
        """
        Implements a Spherical 1x1 Convolution using Conv1D.
        This applies a learned transformation to each pixel independently.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bias (bool): Whether to include bias in the convolution.
        """
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            *args,
            **kwargs,
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Apply 1x1 convolution to each pixel's channel vector.

        Args:
            batch_spherical_image (BatchSphericalImage): Input with ``batch_value``
                of shape (B, N, C_in).

        Returns:
            BatchSphericalImage: Output with ``batch_value`` of shape (B, N, C_out),
                same geometry as input.
        """
        batch_value = batch_spherical_image.batch_value.transpose(-1, -2)  # (B, C, N)
        batch_value = self.conv1d(batch_value)
        batch_value = batch_value.transpose(-1, -2)  # Back to (B, N, C)
        return BatchSphericalImage(
            batch_value=batch_value, vector=batch_spherical_image.vector
        )
