#!/usr/bin/env python3
#
# Created on Sat Feb 22 2025 21:10:13
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.bottleneck import SphericalBottleNeck
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalC3K(nn.Module):
    """
    CSP Bottleneck with 3 convolutions.

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
        """Initialize the SphericalC3K block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function name.
            num_bottlenecks (int): Number of Bottleneck blocks.
            expansion (float, optional): Expansion factor for hidden channels. Defaults to 0.5.
            shortcut (bool, optional): Whether to use a residual shortcut inside each Bottleneck. Defaults to True.
        """
        super().__init__()

        hidden_channels = int(out_channels * expansion)

        self.cbna1 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            conv1x1=True,
        )
        self.cbna2 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            conv1x1=True,
        )
        self.cbna3 = SphericalCBNA(
            in_channels=2 * hidden_channels,
            out_channels=out_channels,
            activation=activation,
            conv1x1=True,
        )

        self.bottleneck_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"bottleneck{i + 1}",
                        SphericalBottleNeck(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            activation=activation,
                            expansion=expansion,
                            shortcut=shortcut,
                            *args,
                            **kwargs,
                        ),
                    )
                    for i in range(num_bottlenecks)
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: two-branch split, bottleneck stack on one, concatenate.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Output with fused branches.
        """
        branch1 = self.cbna1(batch_spherical_image)
        branch2 = self.bottleneck_block(self.cbna2(batch_spherical_image))
        return self.cbna3(concatenate([branch1, branch2]))
