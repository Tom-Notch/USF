#!/usr/bin/env python3
#
# Created on Wed Nov 06 2024 22:22:36
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import math

import torch

from usf.sampler.location.location_sampler import LocationSampler


class Hexahedron(LocationSampler):
    """Location sampler using hexahedron (cube) face subdivision projected onto the sphere."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize hexahedron sampler.

        Args:
            config (dict): Sampler configuration dictionary.
        """
        super().__init__(config, *args, **kwargs)

    def pixel_area2vec(self, average_pixel_area: float) -> torch.Tensor:
        """Generate hexahedron-subdivided vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(self.pixel_area2n_side(average_pixel_area))

    def n_side2vec(self, n_side: int) -> torch.Tensor:
        """Output normalized vectors of the vertices of a hexahedron mapped onto a sphere.

        Args:
            n_side (int): number of subdivisions on a face, note that this will result in different sample density depending on the sampling method

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        # Generate u and v in [-1, 1]
        u = torch.linspace(-1, 1, n_side)
        v = torch.linspace(-1, 1, n_side)
        uu, vv = torch.meshgrid(u, v, indexing="xy")

        # Initialize lists to hold all faces
        x_list = []
        y_list = []
        z_list = []

        # Faces of the hexahedron
        faces = []
        # Top face z = 1
        x = uu
        y = vv
        z = torch.ones_like(uu)
        faces.append((x, y, z))
        # Bottom face z = -1
        x = uu
        y = vv
        z = -torch.ones_like(uu)
        faces.append((x, y, z))
        # Front face x = 1
        x = torch.ones_like(uu)
        y = uu
        z = vv
        faces.append((x, y, z))
        # Back face x = -1
        x = -torch.ones_like(uu)
        y = uu
        z = vv
        faces.append((x, y, z))
        # Right face y = 1
        x = uu
        y = torch.ones_like(uu)
        z = vv
        faces.append((x, y, z))
        # Left face y = -1
        x = uu
        y = -torch.ones_like(uu)
        z = vv
        faces.append((x, y, z))

        # Map hexahedron faces to sphere
        for x, y, z in faces:
            x_sphere = x * torch.sqrt(1 - (y**2) / 2 - (z**2) / 2 + (y**2) * (z**2) / 3)
            y_sphere = y * torch.sqrt(1 - (z**2) / 2 - (x**2) / 2 + (z**2) * (x**2) / 3)
            z_sphere = z * torch.sqrt(1 - (x**2) / 2 - (y**2) / 2 + (x**2) * (y**2) / 3)
            x_list.append(x_sphere.flatten())
            y_list.append(y_sphere.flatten())
            z_list.append(z_sphere.flatten())

        # Combine all points
        x_all = torch.concatenate(x_list)
        y_all = torch.concatenate(y_list)
        z_all = torch.concatenate(z_list)

        # Stack into an array
        all_vectors = torch.vstack((x_all, y_all, z_all)).T

        return all_vectors

    def pixel_area2n_side(self, average_pixel_area: float) -> int:
        """Gets number of subdivisions on a face, n_side, given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians

        Returns:
            int: number of subdivisions on a face
        """
        # average_pixel_area = 4 * torch.pi / (6 * n_side ** 2)
        n_side = math.sqrt(2 * torch.pi / (3 * average_pixel_area))
        return int(n_side)
