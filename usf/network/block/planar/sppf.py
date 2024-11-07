#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 14:38:42
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn

from usf.network.block.planar.cbna import PlanarCBNA


class PlanarSPPF(nn.Module):
    """
    Planar Spatial Pyramid Pooling - Fast (SPPF) layer.

    This layer processes an input tensor as follows:
      1. Applies a 1x1 convolution (`cbna1`) to reduce the input channels to `hidden_channels`,
         where `hidden_channels = in_channels // 2`.
      2. Applies `MaxPool2d` repeatedly (num_pooling_layers times), each time pooling over a `k x k` region.
      3. Concatenates the original output from `cbna1` with each pooled output using `torch.cat`.
      4. Fuses the concatenated features with a final 1x1 convolution (`cbna2`) to produce the output.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        num_pooling_layers: int,
        kernel_size: int = 5,
        global_pool: bool = False,
        expansion: float = 0.5,
    ):
        """Initialize the PlanarSPPF block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function for PlanarCBNA.
            num_pooling_layers (int): Number of MaxPool2d to perform.
            kernel_size (int, optional): Kernel size for MaxPool2d. Defaults to 5.
            global_pool (bool, optional): Whether to include global pooling as an additional branch. Defaults to False.
            expansion (float, optional): Expansion factor to compute hidden_channels.
                                         hidden_channels = int(in_channels * expansion). Defaults to 0.5.
        """
        super().__init__()
        hidden_channels = int(in_channels * expansion)  # Hidden channel count

        # First 1x1 conv: reduces channels to hidden_channels.
        self.cbna1 = PlanarCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

        # Final 1x1 conv: fuses concatenated outputs.
        self.cbna2 = PlanarCBNA(
            in_channels=hidden_channels
            * (num_pooling_layers + (2 if global_pool else 1)),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=activation,
        )

        # MaxPooling layer (stride=1 to keep spatial size)
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        if global_pool:
            self.global_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.num_pooling_layers = num_pooling_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PlanarSPPF module.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        _, _, H, W = x.shape
        y = [self.cbna1(x)]  # Apply first conv
        y.extend(
            self.pool(y[-1]) for _ in range(self.num_pooling_layers)
        )  # Apply max pooling three times
        if hasattr(self, "global_pool"):
            y.append(self.global_pool(y[0]).expand(-1, -1, H, W))
        return self.cbna2(torch.cat(y, dim=1))  # Concatenate and apply final conv
