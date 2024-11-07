#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 13:42:14
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn

from usf.network.block.planar.cbna import PlanarCBNA
from usf.network.layer.planar.sa import PlanarSelfAttention


class PlanarPSA(nn.Module):
    """
    Planar Position-Sensitive Attention Block.

    This block applies self-attention (via `PlanarSelfAttention`) and a feed-forward network
    (using `PlanarCBNA` layers) with optional residual (shortcut) connections.
    """

    def __init__(
        self,
        in_channels: int,
        activation: str | bool,
        num_heads: int,
        shortcut: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the PlanarPSA block.

        Args:
            in_channels (int): Number of input/output channels.
            activation (str | bool): Activation applied to the first `PlanarCBNA` layer.
            num_heads (int): Number of attention heads.
            shortcut (bool, optional): Whether to use residual connections. Defaults to True.
        """
        super().__init__()
        self.shortcut = shortcut

        # Self-Attention Block
        self.attention = PlanarSelfAttention(
            in_channels=in_channels, num_heads=num_heads, *args, **kwargs
        )

        # Feed-Forward Network (Using PlanarCBNA instead of Conv)
        self.feed_forward = nn.Sequential(
            PlanarCBNA(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=activation,
            ),
            PlanarCBNA(
                in_channels=in_channels * 2,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation=False,  # No activation in the last layer
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSA block.

        The block first applies self-attention to update the features. Then, if residual connections are enabled,
        it adds the attention output to the input. Next, the feed-forward network is applied, and again
        a residual connection is optionally added.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        attention_output = self.attention(x)

        if self.shortcut:
            attention_output += x  # Residual connection

        feed_forward_output = self.feed_forward(attention_output)

        if self.shortcut:
            feed_forward_output += attention_output  # Residual connection

        return feed_forward_output
