#!/usr/bin/env python3
#
# Created on Thu Aug 28 2025 16:59:04
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
from usf.utils.torch_numpy import autopad


class PlanarUNet(nn.Module):
    """Planar backbone adopting UNet architecture."""

    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        activation: str,
        kernel_size: int,
        *args,
        **kwargs,
    ):
        """Initialize the PlanarUNet backbone.

        Args:
            in_channels (int): Number of channels of input tensor.
            base_channel (int): The number of base channels for scaling.
            activation (str): Activation type, supports all activation types in PyTorch.
            kernel_size (int): Kernel size for convolutions (must be odd).
        """
        super().__init__()

        padding = autopad(kernel_size, None, 1)
        # activation_layer: nn.Module = getattr(nn, activation)

        self.downsample1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=in_channels,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

        self.downsample2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=2,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

        self.downsample3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=2,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

        self.downsample4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=2,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

        self.neck = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        nn.MaxPool2d(
                            kernel_size=kernel_size,
                            stride=2,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "upconv",
                        PlanarCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                            upsample="interpolation",
                        ),
                    ),
                ]
            )
        )

        self.upsample1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "upconv",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                            upsample="interpolation",
                        ),
                    ),
                ]
            )
        )

        self.upsample2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "upconv",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                            upsample="interpolation",
                        ),
                    ),
                ]
            )
        )

        self.upsample3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "upconv",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                            upsample="interpolation",
                        ),
                    ),
                ]
            )
        )

        self.upsample4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass

        Args:
            x (torch.Tensor): shape (B, C, H, W)

        Returns:
            torch.Tensor: same as x
        """
        # Downsample
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)

        # Neck
        deep_features = self.neck(down4)

        # Upsample
        up1 = self.upsample1(torch.cat([down4, deep_features], dim=1))
        up2 = self.upsample2(torch.cat([down3, up1], dim=1))
        up3 = self.upsample3(torch.cat([down2, up2], dim=1))
        up4 = self.upsample4(torch.cat([down1, up3], dim=1))

        return up4
