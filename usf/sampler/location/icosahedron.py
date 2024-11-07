#!/usr/bin/env python3
#
# Created on Wed Nov 06 2024 22:22:56
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import math

import torch

from usf.sampler.location.location_sampler import LocationSampler


class Icosahedron(LocationSampler):
    """Location sampler using icosahedron face subdivision projected onto the sphere."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize icosahedron sampler.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate icosahedron-subdivided vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(self.pixel_area2n_side(average_pixel_area))

    def n_side2vec(self, n_side: int) -> torch.Tensor:
        """Output normalized vectors of points on a sphere using icosahedron subdivision.

        Args:
            n_side (int): number of subdivisions on a face, note that this will result in different sample density depending on the sampling method

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2

        # Create initial icosahedron vertices
        vertices = torch.Tensor(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ]
        )

        # Normalize the vertices to lie on the unit sphere
        vertices /= torch.linalg.norm(vertices, dim=1)[:, None]

        # Define the 20 faces of the icosahedron (each face is a triangle)
        faces = torch.Tensor(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ]
        ).to(torch.int64)

        # Initialize list to hold all subdivided points
        all_points: list[torch.Tensor] = []

        # Subdivide each triangular face
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Loop over subdivisions
            for i in range(n_side + 1):
                for j in range(n_side - i + 1):
                    k = n_side - i - j
                    # Calculate the barycentric coordinates
                    point = (i * v0 + j * v1 + k * v2) / n_side
                    all_points.append(point)

        # Convert to Torch Tensor
        locations = torch.vstack(all_points)

        # Normalize to unit length
        locations = torch.nn.functional.normalize(locations, p=2, dim=-1)

        return locations

    def pixel_area2n_side(self, average_pixel_area: float) -> int:
        """Gets number of subdivisions on a face, n_side, given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            int: number of subdivisions on a face
        """
        # average_pixel_area \approx 4 * torch.pi / (20 * n_side ** 2 / 2)
        n_side = math.sqrt(2 * torch.pi / (5 * average_pixel_area))
        return int(n_side)
