#!/usr/bin/env python3
#
# Created on Sun Feb 23 2025 21:34:29
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.layer.spherical.sa import SphericalSelfAttention
from usf.utils.spherical_image import BatchSphericalImage


class SphericalPSA(nn.Module):
    """
    Spherical Position-Sensitive Attention Block.

    This block applies spherical self-attention (via SphericalSelfAttention) and a feed-forward network
    (using SphericalCBNA layers) with optional residual (shortcut) connections. It processes a
    BatchSphericalImage and returns a new one with updated batch_value while preserving vector and polar.

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
        """
        Initializes the SphericalPSABlock.

        Args:
            in_channels (int): Number of input/output channels.
            activation (str | bool): activation applied to the first cbna of feed-forward network.
            num_heads (int): Number of attention heads for spherical self-attention.
            shortcut (bool, optional): If True, applies residual shortcut connections. Defaults to True.
        """
        super().__init__()
        self.shortcut = shortcut

        # Spherical self-attention: uses SphericalSelfAttention (which uses nn.MultiheadAttention internally).
        self.attention = SphericalSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            *args,
            **kwargs,
        )

        # Feed-forward network: first layer projects from channels to 2*channels, second layer projects back.
        self.feed_forward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=in_channels,
                            out_channels=in_channels * 2,
                            activation=activation,
                            conv1x1=True,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=in_channels * 2,
                            out_channels=in_channels,
                            activation=False,
                            conv1x1=True,
                        ),
                    ),
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """
        Forward pass of the SphericalPSA.

        The block first applies spherical self-attention to update the features. Then, if residual connections are enabled,
        it adds the attention output to the input; otherwise, it uses the attention output directly. Next, the feed-forward network
        is applied, and again a residual connection is optionally added.

        Args:
            batch_spherical_image (BatchSphericalImage): Input spherical image.

        Returns:
            BatchSphericalImage
        """
        attention_output_batch_spherical_image = self.attention(batch_spherical_image)

        if self.shortcut:
            attention_output_batch_spherical_image.batch_value += (
                batch_spherical_image.batch_value
            )

        feed_forward_batch_spherical_image = self.feed_forward(
            attention_output_batch_spherical_image
        )

        if self.shortcut:
            feed_forward_batch_spherical_image.batch_value += (
                attention_output_batch_spherical_image.batch_value
            )

        return feed_forward_batch_spherical_image
