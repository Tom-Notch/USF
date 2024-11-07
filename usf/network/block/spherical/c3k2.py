#!/usr/bin/env python3
#
# Created on Sun Feb 23 2025 02:19:32
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.c3k import SphericalC3K
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalC3K2(nn.Module):
    """
    Faster CSP Bottleneck with 2 convolutions for spherical data using SphericalC3K blocks as building blocks.

    This block processes a BatchSphericalImage as follows:
      1. It applies a 1x1 convolution (cbna1) to expand the input channels to 2 x hidden_channels.
      2. It then splits the expanded features into two halves (each with hidden_channels channels).
         - The first half (branch1) remains unchanged.
         - The second half is processed sequentially by num_c3k SphericalC3K blocks with 2 bottlenecks.
      3. Independently, the input is also processed by a second 1x1 convolution (cbna2) to yield branch_cbna2.
      4. The outputs from branch1, the transformed second half, and branch_cbna2 are concatenated
         using the provided concatenate function.
      5. A final 1x1 convolution (cbna3) fuses the concatenated features to produce the output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        num_c3k: int,
        expansion: float = 0.5,
        shortcut: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the SphericalC3K2 block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function name to use.
            num_c3k (int): Number of SphericalC3K blocks in the transformation sequence (should be 2).
            expansion (float, optional): Expansion factor to compute hidden channels
                                         (hidden_channels = out_channels x expansion). Defaults to 0.5.
            shortcut (bool, optional): Whether each SphericalC3K block uses residual shortcuts. Defaults to True.
        """
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)

        # Expand input to 2 * hidden_channels.
        self.cbna1 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=2 * self.hidden_channels,
            activation=activation,
            conv1x1=True,
        )

        # Fuse concatenated features from all branches.
        # Total channels after concatenation:
        #   – branch1: hidden_channels (from first half of cbna1 output)
        #   – branch2: hidden_channels (from second half of cbna1 output)
        #   – transformation branch: hidden_channels (output of transformation sequence)
        #   – branch from cbna2: hidden_channels
        # Total = (2 + num_c3k) * hidden_channels.
        self.cbna2 = SphericalCBNA(
            in_channels=(2 + num_c3k) * self.hidden_channels,
            out_channels=out_channels,
            activation=activation,
            conv1x1=True,
        )

        # Build the transformation sequence using num_c3k SphericalC3K blocks.
        self.c3k_block = nn.Sequential(
            OrderedDict(
                (
                    (
                        f"c3k_{i+1}",
                        SphericalC3K(
                            in_channels=self.hidden_channels,
                            out_channels=self.hidden_channels,
                            activation=activation,
                            expansion=expansion,
                            num_bottlenecks=2,  # ! from planar C3K2
                            shortcut=shortcut,
                            *args,
                            **kwargs,
                        ),
                    )
                    for i in range(num_c3k)
                )
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: channel-split, cascade C3k blocks, concatenate all branches.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Output with fused branches.
        """
        branches = self.cbna1(batch_spherical_image).split(
            [self.hidden_channels, self.hidden_channels]
        )

        for c3k in self.c3k_block:
            # convolve last and save
            branches.append(c3k(branches[-1]))

        output = self.cbna2(concatenate(branches))

        self.latest_output_vector = output.vector.clone()

        return output
