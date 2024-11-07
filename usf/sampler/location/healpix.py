#!/usr/bin/env python3
#
# Created on Thu Nov 21 2024 16:39:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import math
from typing import Any

import healpy as hp
import numpy as np
import torch

from usf.sampler.location.location_sampler import LocationSampler


class HEALPix(LocationSampler):
    """Location sampler using the HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) scheme."""

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize HEALPix sampler.

        Args:
            config (dict): Sampler configuration dictionary. Optional
                ``location_sampler_config.nest`` (bool) selects HEALPix nested
                pixel ordering. Defaults to False (ring ordering).
        """
        super().__init__(config, *args, **kwargs)
        self.location_sampler_config: dict[str, Any] = config.get(
            "location_sampler_config", {}
        )
        # HEALPix nest (hierarchical) pixel order: pass nest=True to hp.pix2vec here and
        # the same flag to hp.mollview / hp.cartview / hp.boundaries when visualizing.
        self.nest: bool = self.location_sampler_config.get(
            "nest", self.location_sampler_config.get("nest", False)
        )

    def extra_args(self) -> str:
        """Return extra repr info for HEALPix.

        Returns:
            str: Nest ordering flag.
        """
        return f"nest={self.nest}"

    def pixel_area2vec(
        self, average_pixel_area: float, nest: bool | None = None
    ) -> torch.Tensor:
        """Generate HEALPix vectors matching the target pixel area.

        Args:
            average_pixel_area (float): Desired average area per pixel in steradians.
            nest (bool | None, optional): Override nested pixel ordering. Defaults to instance setting.

        Returns:
            torch.Tensor: Unit Cartesian vectors of shape (N, 3).
        """
        return self.n_side2vec(
            self.pixel_area2n_side(average_pixel_area, nest=nest), nest=nest
        )

    def n_side2vec(self, n_side: int, nest: bool | None = None) -> torch.Tensor:
        """Output normalized vectors of points on a sphere using HEALPix sampling.

        Args:
            n_side (int): number of subdivisions on a face, note that this will result in different sample density depending on the sampling method
            nest (bool | None, optional): whether to use nested ordering. Defaults to None.

        Returns:
            torch.Tensor: torch.Tensor of shape (..., 3) containing co-centric unit 3D vectors (x, y, z), where x pointing forward, y pointing left, and z pointing up
        """
        nest = nest if nest is not None else self.nest

        # Nested (NEST) ordering requires valid hierarchical n_side: power of 2, not larger than 2**30.
        if nest:
            assert (
                n_side & (n_side - 1)
            ) == 0, f"n_side {n_side} must be a power of 2 when nest=True"
            assert n_side <= 2**30, f"n_side {n_side} must be less than 2**30"

        n_pix = hp.nside2npix(n_side)
        # healpy expects NumPy int pixel indices (RING or NEST interpretation via nest=)
        ipix = np.arange(n_pix, dtype=np.int64)
        x, y, z = hp.pix2vec(n_side, ipix, nest=nest)
        vector = torch.from_numpy(np.stack((x, y, z), axis=0)).T.to(dtype=torch.float32)

        return vector

    def pixel_area2n_side(
        self, average_pixel_area: float, nest: bool | None = None
    ) -> int:
        """Gets number of subdivisions on a face, n_side, given average pixel area

        Args:
            average_pixel_area (float): average pixel area in square radians
            nest (bool | None, optional): whether to use nested ordering. Defaults to None.

        Returns:
            int: number of subdivisions on a face
        """
        nest = nest if nest is not None else self.nest

        n_pix = 4 * torch.pi / average_pixel_area

        # cast n_pix to the nearest 12 * n_side ** 2
        n_pix = 12 * math.floor(math.sqrt(n_pix / 12)) ** 2

        n_side = hp.npix2nside(n_pix)
        n_side = int(n_side)

        if nest:
            # Required for NEST: n_side must be a power of 2 (see n_side2vec assertions).
            n_side = 2 ** int(math.floor(math.log2(n_side)))
            assert (
                n_side & (n_side - 1)
            ) == 0, f"n_side {n_side} must be a power of 2 when nest=True"

        return n_side
