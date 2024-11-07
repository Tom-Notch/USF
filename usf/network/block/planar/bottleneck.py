#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 00:53:23
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


class PlanarBottleNeck(nn.Module):
    """Planar bottleneck block with two sequential CBNA layers and optional residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        kernel_sizes: list[int],
        expansion: float = 0.5,
        shortcut: bool = True,
        *args,
        **kwargs,
    ):
        """
        Bottleneck block for planar CNNs with automatic padding to maintain input shape.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function name (must exist in ``torch.nn``).
            kernel_sizes (list[int]): Kernel sizes for the two convolutional layers.
            expansion (float, optional): Expansion factor for the hidden layer channels. Defaults to 0.5.
            shortcut (bool, optional): Whether to use residual shortcut. Defaults to True.
        """
        super().__init__()

        assert expansion > 0.0, "Cannot have negative number of mid channels"
        if shortcut:
            assert (
                in_channels == out_channels
            ), "in_channels must match out_channels, when shortcut is enabled"

        hidden_channels = max(1, int(out_channels * expansion))

        self.shortcut = shortcut

        self.cbna_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=in_channels,
                            out_channels=hidden_channels,
                            kernel_size=kernel_sizes[0],
                            stride=1,
                            activation=activation,
                            *args,
                            **kwargs,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=hidden_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_sizes[1],
                            stride=1,
                            activation=activation,
                            *args,
                            **kwargs,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the bottleneck with optional residual connection.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of shape (B, C_out, H, W).
        """
        conv_output = self.cbna_block(x)
        return x + conv_output if self.shortcut else conv_output
