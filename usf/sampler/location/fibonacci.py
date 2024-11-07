#!/usr/bin/env python3
#
# Created on Fri Jul 25 2025 14:24:04
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


class Fibonacci(LocationSampler):
    """Location sampler that generates points on a sphere using the Fibonacci lattice (golden spiral) method."""

    GOLDEN_RATIO = (1 + 5**0.5) / 2  # ~1.6180339887

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize Fibonacci sampler.

        Args:
            config (dict): Sampler configuration dictionary. Optional
                ``location_sampler_config.golden_ratio`` (float) overrides the default.
        """
        super().__init__(config, *args, **kwargs)
        self.location_sampler_config: dict[str, Any] = config.get(
            "location_sampler_config", {}
        )
        self.golden_ratio = self.location_sampler_config.get(
            "golden_ratio", self.GOLDEN_RATIO
        )

    def extra_args(self) -> str:
        """overridden function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        return f"golden_ratio={self.golden_ratio:.2f}"

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Compute sampling locations given a target average pixel area (in steradians)."""
        return self.n_points2vec(self.pixel_area2n_points(average_pixel_area))

    def pixel_area2n_points(self, average_pixel_area: float) -> int:
        """Estimate the number of sample points for a given average pixel area on the sphere.

        Uses the relation: average_pixel_area ≈ 4π / N, so N ≈ 4π / average_pixel_area.
        Ensures at least 1 point is returned.
        """
        # Compute estimated N (points count) based on sphere area 4π
        if average_pixel_area <= 0:
            raise ValueError("average_pixel_area must be positive")
        N = (4 * math.pi) / average_pixel_area
        N = max(1, int(N))  # ensure at least 1 point
        return N

    def n_points2vec(self, n_points: int) -> torch.Tensor:
        """Generate `n_points` approximately uniformly distributed unit vectors on the sphere.

        Args:
            n_points (int): Number of points to generate on the unit sphere.
        Returns:
            torch.Tensor: Tensor of shape (n_points, 3) with rows being (x, y, z) coordinates on the unit sphere.
        """
        if n_points < 1:
            return torch.empty((0, 3))  # no points

        # Use double precision for intermediate calculations for accuracy
        n = float(n_points)
        indices = torch.arange(n_points, dtype=torch.float32)

        # Golden ratio φ and golden angle (2π/φ)
        theta = 2 * math.pi * indices / self.golden_ratio  # azimuth angles
        # Polar (colatitude) angles via equal-area partitioning
        phi = torch.arccos(1 - 2 * indices / n)

        # Convert to Cartesian coordinates
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        points = torch.stack((x, y, z), dim=1)

        # Cast to float32 for compatibility with other computations
        return points.to(dtype=torch.float32)
