#!/usr/bin/env python3
#
# Created on Sun Feb 16 2025 16:36:20
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch.nn as nn

from usf.utils.spherical_image import BatchSphericalImage


class SphericalBatchNorm(nn.Module):
    """Batch normalization for spherical signals stored as ``BatchSphericalImage``.

    Wraps ``torch.nn.BatchNorm1d`` — the spatial dimension (N pixels) is
    treated as the "length" axis while channels (C) are normalized per-feature.
    Geometry fields (vector, polar, mask) pass through unchanged.
    """

    def __init__(self, num_features: int, *args, **kwargs):
        """
        Spherical Batch Normalization Module

        Args:
            num_features (int): Number of features (i.e., channels `C` in `BatchSphericalImage.batch_value`).
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, *args, **kwargs)

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """
        Forward pass for spherical batch normalization.

        Since `BatchNorm1d` expects input in (B, C, N) format but `BatchSphericalImage.batch_value`
        is in (B, N, C), we transpose before and after applying normalization.
        """
        batch_value = batch_spherical_image.batch_value  # (B, N, C)

        # Apply BatchNorm1d
        normalized_value = self.bn(batch_value.transpose(-1, -2)).transpose(
            -1, -2
        )  # (B, C, N) -> (B, N, C)

        return BatchSphericalImage(
            batch_value=normalized_value,
            vector=batch_spherical_image.vector,
        )
