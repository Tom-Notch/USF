#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Unit tests for usf.utils.spherical_image dataclasses."""

import numpy as np
import pytest
import torch

from usf.utils.spherical_image import (
    BatchSphericalImage,
    SphericalImage,
    concatenate,
    split,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_vector(n: int, dtype: type = np.float32) -> np.ndarray:
    """Generate n roughly uniform unit vectors on the sphere.

    Args:
        n (int): Number of vectors.
        dtype (type): Numpy dtype for the output. Defaults to np.float32.

    Returns:
        np.ndarray: Unit vectors of shape (n, 3).
    """
    rng = np.random.default_rng(0)
    v = rng.standard_normal((n, 3)).astype(dtype)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _make_vector_torch(n: int) -> torch.Tensor:
    """Generate n roughly uniform unit vectors on the sphere as a torch tensor.

    Args:
        n (int): Number of vectors.

    Returns:
        torch.Tensor: Unit vectors of shape (n, 3).
    """
    return torch.from_numpy(_make_vector(n, np.float32))


# ── SphericalImage ───────────────────────────────────────────────────


class TestSphericalImage:
    """Test SphericalImage construction, cloning, splitting, and conversion."""

    def test_init_with_vector_computes_polar(self) -> None:
        """Providing vector should auto-compute polar."""
        vector = _make_vector(10)
        si = SphericalImage(vector=vector)
        assert si.polar is not None
        assert si.polar.shape == (10, 2)

    def test_init_with_polar_computes_vector(self) -> None:
        """Providing polar should auto-compute vector."""
        polar = np.array([[0.0, 0.0], [0.5, 1.0]], dtype=np.float32)
        si = SphericalImage(polar=polar)
        assert si.vector is not None
        assert si.vector.shape == (2, 3)

    def test_init_requires_location(self) -> None:
        """Must provide at least vector or polar."""
        with pytest.raises(ValueError):
            SphericalImage(value=np.zeros((5, 3)))

    def test_value_shape_matches_vector(self) -> None:
        """Value and vector should have the same number of pixels."""
        vector = _make_vector(10)
        value = np.random.rand(10, 3).astype(np.float32)
        si = SphericalImage(value=value, vector=vector)
        assert si.value.shape[0] == si.vector.shape[0]

    def test_default_value_is_colorized(self) -> None:
        """When no value is given, locations should be colorized."""
        vector = _make_vector(10)
        si = SphericalImage(vector=vector)
        assert si.value is not None
        assert si.value.shape == (10, 3)

    def test_clone_is_independent_numpy(self) -> None:
        """Cloned image should be independent from the original."""
        vector = _make_vector(10)
        value = np.ones((10, 3), dtype=np.float32)
        si = SphericalImage(value=value, vector=vector)
        cloned = si.clone()
        cloned.value[:] = 0
        assert np.all(si.value == 1.0)

    def test_copy_is_independent_numpy(self) -> None:
        """Copied image should be independent from the original."""
        vector = _make_vector(10)
        value = np.ones((10, 3), dtype=np.float32)
        si = SphericalImage(value=value, vector=vector)
        copied = si.copy()
        copied.value[:] = 0
        assert np.all(si.value == 1.0)

    def test_to_torch_and_back(self) -> None:
        """to_torch() should convert all arrays to tensors."""
        vector = _make_vector(10)
        value = np.random.rand(10, 3).astype(np.float32)
        si = SphericalImage(value=value, vector=vector)

        si_torch = si.to_torch()
        assert isinstance(si_torch.value, torch.Tensor)
        assert isinstance(si_torch.vector, torch.Tensor)

    def test_split_by_channel(self) -> None:
        """split(n) should divide value channels into n equal parts."""
        vector = _make_vector(10)
        value = np.random.rand(10, 6).astype(np.float32)
        si = SphericalImage(value=value, vector=vector)
        parts = si.split(3)  # split 6 channels into 3 equal parts of 2
        assert len(parts) == 3
        assert all(p.value.shape == (10, 2) for p in parts)

    def test_len(self) -> None:
        """len() should return the number of pixels."""
        vector = _make_vector(20)
        si = SphericalImage(vector=vector)
        assert len(si) == 20


# ── BatchSphericalImage ──────────────────────────────────────────────


class TestBatchSphericalImage:
    """Test BatchSphericalImage construction, indexing, iteration, and cloning."""

    def test_init_with_single_image(self) -> None:
        """Passing a SphericalImage as batch_value should create a batch of 1."""
        vector = _make_vector(10)
        value = np.random.rand(10, 3).astype(np.float32)
        si = SphericalImage(value=value, vector=vector)
        bsi = BatchSphericalImage(batch_value=si)
        assert bsi.batch_value.shape == (1, 10, 3)

    def test_init_with_batch_value(self) -> None:
        """Direct batch_value array should set batch size correctly."""
        vector = _make_vector(10)
        batch_value = np.random.rand(4, 10, 3).astype(np.float32)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        assert bsi.batch_value.shape == (4, 10, 3)
        assert len(bsi) == 4

    def test_getitem_int_returns_spherical_image(self) -> None:
        """Integer indexing should return a SphericalImage."""
        vector = _make_vector(10)
        batch_value = np.random.rand(4, 10, 3).astype(np.float32)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        si = bsi[0]
        assert isinstance(si, SphericalImage)
        assert si.value.shape == (10, 3)

    def test_getitem_slice_returns_batch(self) -> None:
        """Slice indexing should return a sub-BatchSphericalImage."""
        vector = _make_vector(10)
        batch_value = np.random.rand(4, 10, 3).astype(np.float32)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        sub = bsi[1:3]
        assert isinstance(sub, BatchSphericalImage)
        assert len(sub) == 2

    def test_iter_yields_spherical_images(self) -> None:
        """Iterating should yield SphericalImage instances."""
        vector = _make_vector(10)
        batch_value = np.random.rand(3, 10, 3).astype(np.float32)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        items = list(bsi)
        assert len(items) == 3
        assert all(isinstance(si, SphericalImage) for si in items)

    def test_add_concatenates_batches(self) -> None:
        """Adding two BatchSphericalImages should concatenate along the batch dimension."""
        vector = _make_vector(10)
        bv1 = np.random.rand(2, 10, 3).astype(np.float32)
        bv2 = np.random.rand(3, 10, 3).astype(np.float32)
        bsi1 = BatchSphericalImage(batch_value=bv1, vector=vector)
        bsi2 = BatchSphericalImage(batch_value=bv2, vector=vector)
        combined = bsi1 + bsi2
        assert len(combined) == 5

    def test_clone_is_independent(self) -> None:
        """Cloned batch should be independent from the original."""
        vector = _make_vector(10)
        batch_value = np.ones((2, 10, 3), dtype=np.float32)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        cloned = bsi.clone()
        cloned.batch_value[:] = 0
        assert np.all(bsi.batch_value == 1.0)

    def test_to_device(self) -> None:
        """to() should move tensors to the specified device."""
        vector = _make_vector_torch(10)
        batch_value = torch.randn(2, 10, 3)
        bsi = BatchSphericalImage(batch_value=batch_value, vector=vector)
        bsi_cpu = bsi.to(device=torch.device("cpu"))
        assert bsi_cpu.batch_value.device.type == "cpu"


# ── Module-level concatenate / split ─────────────────────────────────


class TestConcatenateSplit:
    """Test module-level concatenate() and split() for spherical images."""

    def test_concatenate_spherical_images(self) -> None:
        """Concatenating along channel dimension should sum channel counts."""
        vector = _make_vector(10)
        si1 = SphericalImage(
            value=np.random.rand(10, 3).astype(np.float32), vector=vector
        )
        si2 = SphericalImage(
            value=np.random.rand(10, 5).astype(np.float32), vector=vector
        )
        merged = concatenate([si1, si2])
        assert merged.value.shape == (10, 8)

    def test_concatenate_batch_spherical_images(self) -> None:
        """Concatenating batch images should sum channel counts per sample."""
        vector = _make_vector(10)
        bsi1 = BatchSphericalImage(
            batch_value=np.random.rand(2, 10, 3).astype(np.float32), vector=vector
        )
        bsi2 = BatchSphericalImage(
            batch_value=np.random.rand(2, 10, 5).astype(np.float32), vector=vector
        )
        merged = concatenate([bsi1, bsi2])
        assert merged.batch_value.shape == (2, 10, 8)

    def test_split_spherical_image(self) -> None:
        """Splitting 6 channels by 3 should yield 3 parts of 2 channels each."""
        vector = _make_vector(10)
        si = SphericalImage(
            value=np.random.rand(10, 6).astype(np.float32), vector=vector
        )
        parts = split(si, 3)
        assert len(parts) == 3
        assert all(p.value.shape == (10, 2) for p in parts)

    def test_split_batch_spherical_image(self) -> None:
        """Splitting at given indices should produce correct channel sizes."""
        vector = _make_vector(10)
        bsi = BatchSphericalImage(
            batch_value=np.random.rand(2, 10, 6).astype(np.float32), vector=vector
        )
        parts = split(bsi, [2, 4])
        assert len(parts) == 3
        assert parts[0].batch_value.shape == (2, 10, 2)
        assert parts[1].batch_value.shape == (2, 10, 2)
        assert parts[2].batch_value.shape == (2, 10, 2)

    def test_concatenate_then_split_roundtrip(self) -> None:
        """concatenate then split should recover the original values."""
        vector = _make_vector(10)
        v1 = np.random.rand(10, 3).astype(np.float32)
        v2 = np.random.rand(10, 5).astype(np.float32)
        si1 = SphericalImage(value=v1, vector=vector)
        si2 = SphericalImage(value=v2, vector=vector)
        merged = concatenate([si1, si2])
        parts = split(merged, [3])
        np.testing.assert_array_equal(parts[0].value, v1)
        np.testing.assert_array_equal(parts[1].value, v2)
