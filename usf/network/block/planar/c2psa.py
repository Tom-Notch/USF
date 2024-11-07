#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 14:12:38
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.planar.cbna import PlanarCBNA
from usf.network.block.planar.psa import PlanarPSA


class PlanarC2PSA(nn.Module):
    """
    Planar C2PSA module with attention mechanism for planar CNNs.

    This module implements a convolutional block with attention as follows:
      1. Applies a 1x1 convolutional block (`cbna1`) to reduce the input channels
         from `in_channels` to `2 x hidden_channels`, where `hidden_channels = int(in_channels x expansion)`.
      2. Splits the resulting tensor along the channel dimension into two equal parts.
      3. Processes the second half through a sequence of `PlanarPSA` blocks (`psa_block`).
      4. Concatenates the unmodified first half with the transformed second half using `torch.cat`.
      5. Applies a final 1x1 convolutional block (`cbna2`) to fuse the concatenated features back to `in_channels`.

    Note:
      - The module uses `in_channels` for both input and output.
      - The number of `PlanarPSA` blocks is controlled by `num_psa`.
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
        Initializes the PlanarC2PSA.

        Args:
            in_channels (int): Number of input (and output) channels.
            activation (str | bool): Activation applied to the CBNA blocks.
            num_psa (int): Number of `PlanarPSA` blocks to stack in the transformation sequence.
            shortcut (bool, optional): If True, applies residual shortcut connections. Defaults to True.
            expansion (float, optional): Expansion factor used to compute `hidden_channels`.
                                         `hidden_channels = int(in_channels x expansion)`. Defaults to 0.5.
        """
        super().__init__()
        self.hidden_channels = int(in_channels * expansion)

        # Expand input channels
        self.cbna1 = PlanarCBNA(
            in_channels=in_channels,
            out_channels=2 * self.hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

        # Final fusion of concatenated features
        self.cbna2 = PlanarCBNA(
            in_channels=2 * self.hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

        # PSA Block Sequence
        self.psa_block = nn.Sequential(
            OrderedDict(
                (
                    (
                        f"psa{i+1}",
                        PlanarPSA(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PlanarC2PSA module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        branch1, branch2 = self.cbna1(x).split(
            [self.hidden_channels, self.hidden_channels], dim=1
        )

        return self.cbna2(torch.cat([branch1, self.psa_block(branch2)], dim=1))
