#!/usr/bin/env python3
#
# Created on Fri Jan 17 2025 14:14:27
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import math

import torch

from usf.sampler.location.location_sampler import LocationSampler


class Tetrahedron(LocationSampler):
    """Location sampler using tetrahedron face subdivision projected onto the sphere."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize tetrahedron sampler.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate tetrahedron-subdivided vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(self.pixel_area2n_side(average_pixel_area))

    def n_side2vec(self, n_side: int) -> torch.Tensor:
        """Output normalized vectors of points on a sphere via barycentric subdivision of a regular Tetrahedron, with one vertex at (1, 0, 0).

        Args:
            n_side (int): number of subdivisions on each face.

        Returns:
            torch.Tensor: shape (N, 3) co-centric unit vectors on the sphere.
        """
        # A regular tetrahedron can be placed so that one vertex is at (1,0,0).
        # The other 3 vertices each have dot = -1/3 with (1,0,0) => angle ~ 109.47 deg
        # One possible set of coordinates (all are unit length):
        v0 = torch.Tensor([1.0, 0.0, 0.0])  # "forward" vertex
        v1 = torch.Tensor([-1 / 3, 2 * math.sqrt(2) / 3, 0.0])
        v2 = torch.Tensor([-1 / 3, -math.sqrt(2) / 3, math.sqrt(6) / 3])
        v3 = torch.Tensor([-1 / 3, -math.sqrt(2) / 3, -math.sqrt(6) / 3])

        # Collect into an array
        vertices = torch.vstack([v0, v1, v2, v3])  # shape (4,3)

        # Define the faces by indices into 'vertices'
        # Each face is a triangle => 4 faces total
        faces = torch.Tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]).to(
            torch.int64
        )

        # Barycentric subdivision of each face
        all_points: list[torch.Tensor] = []
        for f in faces:
            A = vertices[f[0]]
            B = vertices[f[1]]
            C = vertices[f[2]]

            for i in range(n_side + 1):
                for j in range(n_side - i + 1):
                    k = n_side - i - j
                    # Barycentric combination
                    p = (i * A + j * B + k * C) / n_side
                    all_points.append(p)

        locations = torch.stack(all_points, dim=0)

        # rotate +pi/2 along x axis to make one edge up
        theta = torch.tensor(
            math.pi / 2, dtype=locations.dtype, device=locations.device
        )
        cos, sin = torch.cos(theta), torch.sin(theta)
        R = torch.tensor(
            [
                [1, 0, 0],
                [0, cos, -sin],
                [0, sin, cos],
            ],
            dtype=locations.dtype,
            device=locations.device,
        )
        locations = locations @ R.T

        # Normalize to unit length
        locations = torch.nn.functional.normalize(locations, p=2, dim=-1)

        return locations

    def pixel_area2n_side(self, average_pixel_area: float) -> int:
        """Approximate formula for n_side given a desired average_pixel_area on the sphere.

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            int: number of subdivisions on a face
        """
        # average_pixel_area = 4 * π / (4 * n_side ** 2 / 2)
        n_side = math.sqrt(2.0 * torch.pi / average_pixel_area)
        return int(n_side)
