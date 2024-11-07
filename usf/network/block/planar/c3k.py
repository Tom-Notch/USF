#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 00:55:16
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.planar.bottleneck import PlanarBottleNeck
from usf.network.block.planar.cbna import PlanarCBNA


class PlanarC3K(nn.Module):
    """
    CSP Bottleneck with 3 convolutions for planar CNNs.

    This block splits the input into two branches:
      - Branch 1: Applies a 1x1 convolution (cbna1) followed by a sequence of Bottleneck blocks (bottleneck_block).
      - Branch 2: Applies a 1x1 convolution (cbna2) directly.
    Their outputs are concatenated along the channel dimension and fused with a final 1x1 convolution (cbna3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        num_bottlenecks: int,
        expansion: float = 0.5,
        shortcut: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the PlanarC3K block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function.
            num_bottlenecks (int): Number of Bottleneck blocks.
            expansion (float, optional): Expansion factor used to calculate hidden channels. hidden_channels = out_channels x expansion. Defaults to 0.5.
            shortcut (bool, optional): Whether to use a residual shortcut inside each Bottleneck. Defaults to True.
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        self.cbna1 = PlanarCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )
        self.cbna2 = PlanarCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )
        self.cbna3 = PlanarCBNA(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

        self.bottleneck_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"bottleneck{i + 1}",
                        PlanarBottleNeck(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            expansion=expansion,
                            shortcut=shortcut,
                            activation=activation,
                            *args,
                            **kwargs,
                        ),
                    )
                    for i in range(num_bottlenecks)
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: split into two branches, apply bottleneck stack to one, concatenate.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of shape (B, C_out, H, W).
        """
        branch1 = self.cbna1(x)
        branch2 = self.bottleneck_block(self.cbna2(x))
        return self.cbna3(torch.cat([branch1, branch2], dim=1))
