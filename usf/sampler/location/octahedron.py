#!/usr/bin/env python3
#
# Created on Wed Nov 06 2024 22:22:47
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import math

import torch

from usf.sampler.location.location_sampler import LocationSampler


class Octahedron(LocationSampler):
    """Location sampler using octahedron face subdivision projected onto the sphere."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize octahedron sampler.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate octahedron-subdivided vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(self.pixel_area2n_side(average_pixel_area))

    def n_side2vec(self, n_side: int) -> torch.Tensor:
        """Output normalized vectors of points on a sphere using octahedron subdivision.
        #! Note: the sample point is non-uniformly distributed, might need further investigation on subdivision method on each triangle

        Args:
            n_side (int): number of subdivisions on a face, note that this will result in different sample density depending on the sampling method

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        # Create initial octahedron vertices for the front-top-left face
        v0 = torch.Tensor([1, 0, 0])  # front vertex
        v2 = torch.Tensor([0, 1, 0])  # left vertex
        v1 = torch.Tensor([0, 0, 1])  # top vertex

        # Initialize list to hold all subdivided points
        face_points = []

        # Subdivide the triangle
        for i in range(n_side + 1):
            for j in range(n_side - i + 1):
                k = n_side - i - j
                # Calculate the barycentric coordinates
                point = (i * v0 + j * v1 + k * v2) / n_side
                face_points.append(point)

        # in shape (3, N)
        all_points = torch.stack(face_points, dim=0).T

        # 90-degree rotation around z-dim
        R_z_90 = torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        all_points = torch.hstack((all_points, R_z_90 @ all_points))

        # 180-degree rotation around z-dim
        R_z_180 = torch.Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        all_points = torch.hstack((all_points, R_z_180 @ all_points))

        # 180-degree rotation around x-dim
        R_x_180 = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        all_points = torch.hstack((all_points, R_x_180 @ all_points))

        # turn to shape (N, 3)
        all_points = all_points.T

        # Normalize to unit length
        all_points = torch.nn.functional.normalize(all_points, p=2, dim=-1)

        return all_points

    def pixel_area2n_side(self, average_pixel_area: float) -> int:
        """Gets number of subdivisions on a face, n_side, given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            int: number of subdivisions on a face
        """
        # average_pixel_area = 4 * torch.pi / (8 * n_side ** 2 / 2)
        n_side = math.sqrt(torch.pi / average_pixel_area)
        return int(n_side)
