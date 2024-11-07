#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Unit tests for usf.utils.spherical coordinate conversions and distances."""

import numpy as np
import pytest
import torch

from usf.utils.spherical import (
    cartesian2polar,
    normalize_cartesian,
    normalize_polar,
    polar2cartesian,
    spherical_distance,
)

# ── polar2cartesian / cartesian2polar round-trip ─────────────────────


class TestPolar2Cartesian:
    """Test polar <-> cartesian conversions for both numpy and torch."""

    KNOWN_POINTS = [
        # (theta, phi) -> (x, y, z)
        (0.0, 0.0, 1.0, 0.0, 0.0),  # front: theta=0, phi=0 -> (1, 0, 0)
        (np.pi / 2, 0.0, 0.0, 0.0, 1.0),  # north pole: theta=pi/2 -> (0, 0, 1)
        (-np.pi / 2, 0.0, 0.0, 0.0, -1.0),  # south pole: theta=-pi/2 -> (0, 0, -1)
        (0.0, np.pi / 2, 0.0, 1.0, 0.0),  # left: phi=pi/2 -> (0, 1, 0)
        (0.0, -np.pi / 2, 0.0, -1.0, 0.0),  # right: phi=-pi/2 -> (0, -1, 0)
    ]

    @pytest.mark.parametrize("theta,phi,ex,ey,ez", KNOWN_POINTS)
    def test_known_points_numpy(
        self, theta: float, phi: float, ex: float, ey: float, ez: float
    ) -> None:
        """Verify known polar -> cartesian mappings with numpy.

        Args:
            theta (float): Latitude in radians.
            phi (float): Longitude in radians.
            ex (float): Expected x component.
            ey (float): Expected y component.
            ez (float): Expected z component.
        """
        polar = np.array([[theta, phi]], dtype=np.float32)
        cart = polar2cartesian(polar)
        expected = np.array([[ex, ey, ez]], dtype=np.float32)
        np.testing.assert_allclose(cart, expected, atol=1e-6)

    @pytest.mark.parametrize("theta,phi,ex,ey,ez", KNOWN_POINTS)
    def test_known_points_torch(
        self, theta: float, phi: float, ex: float, ey: float, ez: float
    ) -> None:
        """Verify known polar -> cartesian mappings with torch.

        Args:
            theta (float): Latitude in radians.
            phi (float): Longitude in radians.
            ex (float): Expected x component.
            ey (float): Expected y component.
            ez (float): Expected z component.
        """
        polar = torch.tensor([[theta, phi]], dtype=torch.float32)
        cart = polar2cartesian(polar)
        expected = torch.tensor([[ex, ey, ez]], dtype=torch.float32)
        torch.testing.assert_close(cart, expected, atol=1e-6, rtol=1e-5)

    def test_round_trip_numpy(self, rng_np: np.random.Generator) -> None:
        """polar -> cartesian -> polar should be identity (up to normalization).

        Args:
            rng_np (np.random.Generator): Seeded random generator.
        """
        theta = rng_np.uniform(-np.pi / 2, np.pi / 2, size=(100, 1)).astype(np.float64)
        phi = rng_np.uniform(-np.pi, np.pi, size=(100, 1)).astype(np.float64)
        polar = np.hstack([theta, phi])

        recovered = cartesian2polar(polar2cartesian(polar))
        np.testing.assert_allclose(recovered, polar, atol=1e-6)

    def test_round_trip_torch(self) -> None:
        """polar -> cartesian -> polar should be identity (up to normalization)."""
        torch.manual_seed(42)
        theta = torch.rand(100, 1, dtype=torch.float64) * np.pi - np.pi / 2
        phi = torch.rand(100, 1, dtype=torch.float64) * 2 * np.pi - np.pi
        polar = torch.cat([theta, phi], dim=1)

        recovered = cartesian2polar(polar2cartesian(polar))
        torch.testing.assert_close(recovered, polar, atol=1e-6, rtol=1e-5)

    def test_output_is_unit_vectors_numpy(self, rng_np: np.random.Generator) -> None:
        """All output vectors should have unit norm.

        Args:
            rng_np (np.random.Generator): Seeded random generator.
        """
        polar = rng_np.uniform(-1, 1, size=(50, 2)).astype(np.float32)
        cart = polar2cartesian(polar)
        norms = np.linalg.norm(cart, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_output_is_unit_vectors_torch(self) -> None:
        """All output vectors should have unit norm."""
        polar = torch.randn(50, 2, dtype=torch.float32)
        cart = polar2cartesian(polar)
        norms = torch.linalg.norm(cart, dim=-1)
        torch.testing.assert_close(norms, torch.ones(50), atol=1e-6, rtol=1e-5)

    def test_batch_shape_preserved_numpy(self) -> None:
        """Batch dimensions should be preserved through conversion."""
        polar = np.zeros((2, 3, 2), dtype=np.float32)
        cart = polar2cartesian(polar)
        assert cart.shape == (2, 3, 3)

    def test_batch_shape_preserved_torch(self) -> None:
        """Batch dimensions should be preserved through conversion."""
        polar = torch.zeros(2, 3, 2, dtype=torch.float32)
        cart = polar2cartesian(polar)
        assert cart.shape == (2, 3, 3)


# ── normalize_cartesian ──────────────────────────────────────────────


class TestNormalizeCartesian:
    """Test normalize_cartesian for both numpy and torch."""

    def test_already_unit_numpy(self) -> None:
        """Unit vectors should be unchanged."""
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        out = normalize_cartesian(v)
        np.testing.assert_allclose(out, v, atol=1e-7)

    def test_scales_to_unit_torch(self) -> None:
        """Non-unit vectors should be scaled to unit norm."""
        v = torch.tensor([[3.0, 0.0, 4.0]], dtype=torch.float32)
        out = normalize_cartesian(v)
        expected = torch.tensor([[0.6, 0.0, 0.8]], dtype=torch.float32)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


# ── normalize_polar ──────────────────────────────────────────────────


class TestNormalizePolar:
    """Test normalize_polar wrapping behavior."""

    def test_already_canonical_numpy(self) -> None:
        """Already-canonical polar coordinates should be unchanged."""
        polar = np.array([[0.5, 1.0]], dtype=np.float32)
        out = normalize_polar(polar)
        np.testing.assert_allclose(out, polar, atol=1e-7)

    def test_wraps_phi_torch(self) -> None:
        """phi > pi should wrap to [-pi, pi]."""
        polar = torch.tensor([[0.0, np.pi + 0.5]], dtype=torch.float64)
        out = normalize_polar(polar)
        assert -np.pi <= out[0, 1].item() <= np.pi


# ── spherical_distance ───────────────────────────────────────────────


class TestSphericalDistance:
    """Test geodesic distance computation for both numpy and torch."""

    def test_same_point_is_zero_numpy(self) -> None:
        """Distance from a point to itself should be zero."""
        p = np.array([[0.3, 0.7]], dtype=np.float64)
        d = spherical_distance(p, p)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_same_point_is_zero_torch(self) -> None:
        """Distance from a point to itself should be zero."""
        p = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
        d = spherical_distance(p, p)
        torch.testing.assert_close(
            d, torch.tensor([0.0], dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_antipodal_is_pi_numpy(self) -> None:
        """Distance between north and south pole should be pi."""
        north = np.array([[np.pi / 2, 0.0]], dtype=np.float64)
        south = np.array([[-np.pi / 2, 0.0]], dtype=np.float64)
        d = spherical_distance(north, south)
        np.testing.assert_allclose(d, np.pi, atol=1e-10)

    def test_antipodal_is_pi_torch(self) -> None:
        """Distance between north and south pole should be pi."""
        north = torch.tensor([[np.pi / 2, 0.0]], dtype=torch.float64)
        south = torch.tensor([[-np.pi / 2, 0.0]], dtype=torch.float64)
        d = spherical_distance(north, south)
        torch.testing.assert_close(
            d, torch.tensor([np.pi], dtype=torch.float64), atol=1e-10, rtol=0.0
        )

    def test_quarter_sphere_numpy(self) -> None:
        """Front to left should be pi/2."""
        front = np.array([[0.0, 0.0]], dtype=np.float64)
        left = np.array([[0.0, np.pi / 2]], dtype=np.float64)
        d = spherical_distance(front, left)
        np.testing.assert_allclose(d, np.pi / 2, atol=1e-10)

    def test_accepts_cartesian_input_numpy(self) -> None:
        """spherical_distance should accept 3D cartesian vectors too."""
        p1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        p2 = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        d = spherical_distance(p1, p2)
        np.testing.assert_allclose(d, np.pi / 2, atol=1e-10)

    def test_symmetry_torch(self) -> None:
        """d(a, b) == d(b, a)."""
        torch.manual_seed(7)
        a = torch.randn(10, 3, dtype=torch.float64)
        b = torch.randn(10, 3, dtype=torch.float64)
        d_ab = spherical_distance(a, b)
        d_ba = spherical_distance(b, a)
        torch.testing.assert_close(d_ab, d_ba, atol=1e-12, rtol=0.0)

    def test_non_negative_numpy(self, rng_np: np.random.Generator) -> None:
        """All distances should be non-negative.

        Args:
            rng_np (np.random.Generator): Seeded random generator.
        """
        p1 = rng_np.standard_normal((50, 3)).astype(np.float64)
        p2 = rng_np.standard_normal((50, 3)).astype(np.float64)
        d = spherical_distance(p1, p2)
        assert np.all(d >= -1e-12)


# ── numpy / torch consistency ────────────────────────────────────────


class TestNumpyTorchConsistency:
    """Verify numpy and torch implementations produce the same results."""

    def test_polar2cartesian_matches(self) -> None:
        """Numpy and torch polar2cartesian should agree to machine precision."""
        polar_np = np.array([[0.3, -1.2], [1.0, 2.5]], dtype=np.float64)
        polar_pt = torch.from_numpy(polar_np)
        cart_np = polar2cartesian(polar_np)
        cart_pt = polar2cartesian(polar_pt)
        np.testing.assert_allclose(cart_np, cart_pt.numpy(), atol=1e-12)

    def test_cartesian2polar_matches(self) -> None:
        """Numpy and torch cartesian2polar should agree to machine precision."""
        v_np = np.array([[0.5, 0.3, 0.8], [-0.2, 0.9, -0.4]], dtype=np.float64)
        v_pt = torch.from_numpy(v_np)
        polar_np = cartesian2polar(v_np)
        polar_pt = cartesian2polar(v_pt)
        np.testing.assert_allclose(polar_np, polar_pt.numpy(), atol=1e-12)

    def test_spherical_distance_matches(self) -> None:
        """Numpy and torch spherical_distance should agree to machine precision."""
        p1_np = np.array([[0.1, 0.5], [-0.3, 2.0]], dtype=np.float64)
        p2_np = np.array([[0.8, -1.0], [0.2, 0.3]], dtype=np.float64)
        p1_pt = torch.from_numpy(p1_np)
        p2_pt = torch.from_numpy(p2_np)
        d_np = spherical_distance(p1_np, p2_np)
        d_pt = spherical_distance(p1_pt, p2_pt)
        np.testing.assert_allclose(d_np, d_pt.numpy(), atol=1e-12)
