#!/usr/bin/env python3
#
# Created on Sun Feb 23 2025 21:59:08
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.block.spherical.psa import SphericalPSA
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalC2PSA(nn.Module):
    """
    Spherical C2PSA module with attention mechanism for spherical data.

    This module implements a spherical convolutional block with attention as follows:
      1. Applies a 1x1 spherical convolution block (cbna1) to reduce the input channels
         from in_channels to 2 x hidden_channels, where hidden_channels = int(in_channels x expansion).
      2. Splits the resulting BatchSphericalImage along the channel dimension into two equal parts.
      3. Processes the second half through a sequence of SphericalPSA blocks (psa_block).
      4. Concatenates the unmodified first half with the transformed second half using the provided concatenate function.
      5. Applies a final 1x1 spherical convolution block (cbna2) to fuse the concatenated features back to in_channels.

    Note:
      - The module uses `in_channels` for both input and output.
      - The number of `SphericalPSA` blocks is controlled by `num_psa`.
    """

    def __init__(
        self,
        in_channels: int,
        activation: str | bool,
        num_psa: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        Initializes the SphericalC2PSA.

        Args:
            in_channels (int): Number of input (and output) channels.
            activation (str | bool): activation applied to the cbna blocks.
            num_psa (int): Number of SphericalPSA blocks to stack in the transformation sequence.
            shortcut (bool, optional): If True, applies residual shortcut connections. Defaults to True.
            expansion (float, optional): Expansion factor used to compute hidden_channels.
                                          hidden_channels = int(in_channels x expansion). Defaults to 0.5.
        """
        super().__init__()
        self.hidden_channels = int(in_channels * expansion)

        self.cbna1 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=2 * self.hidden_channels,
            activation=activation,
            conv1x1=True,
        )

        self.cbna2 = SphericalCBNA(
            in_channels=2 * self.hidden_channels,
            out_channels=in_channels,
            activation=activation,
            conv1x1=True,
        )

        self.psa_block = nn.Sequential(
            OrderedDict(
                (
                    (
                        f"psa{i+1}",
                        SphericalPSA(
                            in_channels=self.hidden_channels,
                            activation=activation,
                            shortcut=shortcut,
                            num_heads=max(1, self.hidden_channels // 64),
                            *args,
                            **kwargs,
                        ),
                    )
                    for i in range(num_psa)
                )
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """
        Forward pass of the SphericalC2PSA module.

        Args:
            batch_spherical_image (BatchSphericalImage): Input spherical image.

        Returns:
            BatchSphericalImage
        """
        branch1, branch2 = self.cbna1(batch_spherical_image).split(
            [self.hidden_channels, self.hidden_channels]
        )
        return self.cbna2(concatenate([branch1, self.psa_block(branch2)]))
