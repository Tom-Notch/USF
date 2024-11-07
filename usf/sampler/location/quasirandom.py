#!/usr/bin/env python3
#
# Created on Tue Jul 29 2025 16:32:15
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import math
from typing import Any

import torch

from usf.sampler.location.location_sampler import LocationSampler


class QuasiRandom(LocationSampler):
    """
    Location sampler that generates approximately uniform points on the sphere using
    a 2D quasi-random (low-discrepancy) sequence and Lambert equal-area projection.
    Source: https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    PLASTIC_RATIO = 1.32471795724474602596  # Plastic ratio: https://en.wikipedia.org/wiki/Plastic_ratio

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize quasi-random sampler with R-sequence parameters.

        Args:
            config (dict): Sampler configuration dictionary. Optional keys under
                ``location_sampler_config``: ``plastic_ratio``, ``alpha_u``,
                ``alpha_v``, ``s0_u``, ``s0_v``.
        """
        super().__init__(config, *args, **kwargs)
        # Golden ratio increments for R-sequence (good default)
        self.location_sampler_config: dict[str, Any] = config.get(
            "location_sampler_config", {}
        )
        self.plastic_ratio = self.location_sampler_config.get(
            "plastic_ratio", self.PLASTIC_RATIO
        )
        self.alpha_u = self.location_sampler_config.get(
            "alpha_u", 1.0 / self.plastic_ratio
        )  # ≈ 0.6180...
        self.alpha_v = self.location_sampler_config.get(
            "alpha_v", 1.0 / self.plastic_ratio**2
        )  # ≈ 0.7320...
        self.s0_u = self.location_sampler_config.get("s0_u", 0.5)  # start offset for u
        self.s0_v = self.location_sampler_config.get("s0_v", 0.5)  # start offset for v

    def extra_args(self) -> str:
        """overridden function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        return (
            f"plastic_ratio={self.plastic_ratio:.2f}"
            f", alpha_u={self.alpha_u:.2f}"
            f", alpha_v={self.alpha_v:.2f}"
            f", s0_u={self.s0_u:.2f}"
            f", s0_v={self.s0_v:.2f}"
        )

    def n_points2vec(self, n_points: int) -> torch.Tensor:
        """Generate ``n_points`` quasi-random vectors on the sphere via Lambert equal-area mapping.

        Args:
            n_points (int): Number of sample points.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3), dtype float32.
        """
        # float64 is critical: the multiplicative form (s0 + idx * alpha) % 1.0
        # needs ~10 fractional digits to avoid collisions after dedup.
        # In float32, idx * alpha at idx=460K has ULP ~0.03, destroying the fractional part.
        idx = torch.arange(n_points, dtype=torch.float64)
        u = (self.s0_u + idx * self.alpha_u) % 1.0
        v = (self.s0_v + idx * self.alpha_v) % 1.0

        # Lambert equal-area mapping to sphere
        phi = 2 * math.pi * v  # longitude
        z = 1 - 2 * u  # latitude (cosine)
        r = torch.sqrt(1 - z**2)
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        pts = torch.stack((x, y, z), dim=1)

        return pts.to(dtype=torch.float32)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate quasi-random vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_points2vec(self.pixel_area2n_points(average_pixel_area))

    def pixel_area2n_points(self, average_pixel_area: float) -> int:
        """Estimate the number of sample points for a given average pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            int: Estimated number of points (at least 1).
        """
        # Sphere area = 4pi, so N ~ 4pi / area
        N_est = (4 * math.pi) / float(average_pixel_area)
        return max(1, int(N_est))
