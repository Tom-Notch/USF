#!/usr/bin/env python3
#
# Created on Wed Nov 06 2024 22:22:20
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import math

import torch

from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.spherical import polar2cartesian


class Equirectangular(LocationSampler):
    """Location sampler using equirectangular (latitude-longitude) grid on the sphere."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize equirectangular sampler.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate equirectangular grid vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(self.pixel_area2n_side(average_pixel_area))

    def n_side2vec(self, n_side: int) -> torch.Tensor:
        """Output normalized vectors of points on a sphere using equirectangular sampling.

        Args:
            n_side (int): number of subdivisions on a face, note that this will result in different sample density depending on the sampling method

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        # Number of latitude and longitude points
        n_lat = n_side + 1  # Including the poles
        n_lon = n_side * 2  # To maintain approximately square grid cells

        # Generate latitude values from -90 to +90 degrees in radians
        latitudes = torch.linspace(-torch.pi / 2, torch.pi / 2, n_lat)

        # Initialize list to hold all points
        all_points: list[torch.Tensor] = []

        for theta in latitudes:
            # For each latitude, generate longitude values
            # Adjust the number of longitude points at the poles to avoid duplication
            if torch.isclose(
                torch.abs(theta),
                torch.tensor(torch.pi / 2, dtype=theta.dtype, device=theta.device),
            ):
                longitudes = torch.tensor([0.0])
            else:
                # Compute the last value as pi - (2*pi/n_lon)
                longitudes = torch.linspace(
                    -torch.pi, torch.pi - 2 * torch.pi / n_lon, n_lon
                )

            # Create an array of [theta, phi] pairs
            polar = torch.hstack(
                (torch.full_like(longitudes, theta)[:, None], longitudes[:, None])
            )
            all_points.extend(polar2cartesian(polar))

        # Convert to Torch Tensor
        locations = torch.vstack(all_points)

        return locations

    def pixel_area2n_side(self, average_pixel_area: float) -> int:
        """Gets number of subdivisions on a face, n_side, given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            int: number of subdivisions on a face
        """
        # average_pixel_area = 4 * torch.pi / (2 * n_side ** 2)
        n_side = math.sqrt(2 * torch.pi / average_pixel_area)
        return int(n_side)
