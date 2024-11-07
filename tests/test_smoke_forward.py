#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Smoke tests: instantiate each model and run a single forward pass on CPU with dummy data."""

import numpy as np
import pytest
import torch

from usf.utils.spherical_image import BatchSphericalImage


def _make_batch_spherical_image(
    batch_size: int, n_pixels: int, channels: int
) -> BatchSphericalImage:
    """Create a dummy BatchSphericalImage on CPU.

    Args:
        batch_size (int): Number of images in the batch.
        n_pixels (int): Number of pixels per image.
        channels (int): Number of value channels per pixel.

    Returns:
        BatchSphericalImage: Dummy batch spherical image with random unit vectors and values.
    """
    rng = np.random.default_rng(0)
    vector = rng.standard_normal((n_pixels, 3)).astype(np.float32)
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)
    batch_value = torch.randn(batch_size, n_pixels, channels)
    return BatchSphericalImage(batch_value=batch_value, vector=torch.from_numpy(vector))


class TestPlanarMnistForward:
    """Forward pass tests for the planar MNIST model."""

    def test_forward_shape(self) -> None:
        """Output should be (B, 10) for 10-class classification."""
        from usf.network.model.mnist import PlanarMnist

        model = PlanarMnist()
        x = torch.randn(2, 1, 28, 28)  # (B, C, H, W)
        out = model(x)
        assert out.shape == (2, 10)

    def test_output_is_finite(self) -> None:
        """All output values should be finite (no NaN or Inf)."""
        from usf.network.model.mnist import PlanarMnist

        model = PlanarMnist()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert torch.isfinite(out).all()


class TestPlanarSemanticSegmentationForward:
    """Forward pass tests for the planar semantic segmentation model."""

    @pytest.mark.parametrize("backbone", ["DeepLabV3", "UNet"])
    def test_forward_shape(self, backbone: str) -> None:
        """Output should have shape (B, num_categories, ...).

        Args:
            backbone (str): Backbone architecture name.
        """
        from usf.network.model.semantic_segmentation import (
            PlanarSemanticSegmentation,
        )

        model = PlanarSemanticSegmentation(
            backbone=backbone,
            num_categories=14,
            in_channels=3,
            base_channel=16,
            activation="ReLU",
            expansion=0.5,
            kernel_size=3,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)  # (B, C, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 14  # num_categories


class TestPlanarObjectDetectionForward:
    """Forward pass tests for the planar object detection model."""

    def test_forward_returns_dict(self) -> None:
        """Output should be a dict containing heatmap tensors."""
        from usf.network.model.object_detection import PlanarObjectDetection

        model = PlanarObjectDetection(
            num_categories=10,
            in_channels=3,
            base_channel=16,
            activation="ReLU",
            kernel_size=3,
            expansion=0.5,
            num_psa=1,
            attention_backend="plain",
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, dict)
