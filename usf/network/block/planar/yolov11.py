#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 16:35:49
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.planar.c2psa import PlanarC2PSA
from usf.network.block.planar.c3k2 import PlanarC3K2
from usf.network.block.planar.cbna import PlanarCBNA
from usf.network.block.planar.sppf import PlanarSPPF


class PlanarYOLOv11(nn.Module):
    """Planar task-agnostic backbone adopting YOLO v11 architecture."""

    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        activation: str,
        multilevel: bool,
        expansion: float,
        kernel_size: int,
        num_psa: int,
        attention_backend: str,
        *args,
        **kwargs,
    ):
        """Initialize the PlanarYOLOv11 backbone.

        Args:
            in_channels (int): Number of channels of input tensor.
            base_channel (int): The number of base channels for scaling.
            activation (str): Activation type, supports all activation types in PyTorch.
            multilevel (bool): Whether to output multilevel features.
            expansion (float): Expansion ratio for hidden layers.
            kernel_size (int): Kernel size for convolutions (must be odd).
            num_psa (int): Number of PSA attention blocks inside C2PSA at deep neck.
            attention_backend (str): Attention backend to use.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be an odd number."

        self.downsample1 = PlanarCBNA(
            in_channels=in_channels,
            out_channels=base_channel * 2**0,
            activation=activation,
            kernel_size=kernel_size,
            stride=2,
        )

        self.downsample2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        PlanarCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    (
                        "c3k2",
                        PlanarC3K2(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_sizes=[kernel_size, kernel_size],
                            num_c3k=3,
                            expansion=expansion,
                            shortcut=False,
                        ),
                    ),
                ]
            )
        )

        self.downsample3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    (
                        "c3k2",
                        PlanarC3K2(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            kernel_sizes=[kernel_size, kernel_size],
                            num_c3k=6,
                            expansion=expansion,
                            shortcut=False,
                        ),
                    ),
                ]
            )
        )

        self.downsample4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    (
                        "c3k2",
                        PlanarC3K2(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            kernel_sizes=[kernel_size, kernel_size],
                            num_c3k=6,
                            expansion=expansion,
                            shortcut=True,
                        ),
                    ),
                ]
            )
        )

        self.downsample5 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        PlanarCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=2,
                        ),
                    ),
                    (
                        "c3k2",
                        PlanarC3K2(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            kernel_sizes=[kernel_size, kernel_size],
                            num_c3k=3,
                            expansion=expansion,
                            shortcut=True,
                        ),
                    ),
                ]
            )
        )

        # Neck
        self.neck = nn.Sequential(
            OrderedDict(
                [
                    (
                        "sppf",
                        PlanarSPPF(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            num_pooling_layers=3,  # from YOLOv11
                            kernel_size=5,  # from YOLOv11
                            expansion=expansion,
                        ),
                    ),
                    (
                        "c2psa",
                        PlanarC2PSA(
                            in_channels=base_channel * 2**4,
                            activation=activation,
                            num_psa=num_psa,
                            shortcut=True,
                            expansion=expansion,
                            backend=attention_backend,
                        ),
                    ),
                ]
            )
        )

        # Upsample Stage 1
        self.upsample1 = PlanarCBNA(
            in_channels=base_channel * 2**4,
            out_channels=base_channel * 2**3,
            activation=activation,
            kernel_size=kernel_size,
            stride=2,
            upsample="interpolation",
        )
        self.c3k2_upsample1 = PlanarC3K2(
            in_channels=base_channel * 2**4,
            out_channels=base_channel * 2**3,
            activation=activation,
            kernel_sizes=[kernel_size, kernel_size],
            num_c3k=3,
            expansion=expansion,
            shortcut=True,
        )

        # Upsample Stage 2
        self.upsample2 = PlanarCBNA(
            in_channels=base_channel * 2**3,
            out_channels=base_channel * 2**2,
            activation=activation,
            kernel_size=kernel_size,
            stride=2,
            upsample="interpolation",
        )
        self.c3k2_upsample2 = PlanarC3K2(
            in_channels=base_channel * 2**3,
            out_channels=base_channel * 2**2,
            activation=activation,
            kernel_sizes=[kernel_size, kernel_size],
            num_c3k=3,
            expansion=expansion,
            shortcut=True,
        )

        self.multilevel = multilevel
        if self.multilevel:
            # Additional downsampling stages for multi-level outputs
            self.downsample6 = PlanarCBNA(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**3,
                activation=activation,
                kernel_size=kernel_size,
                stride=2,
            )
            self.c3k2_downsample6 = PlanarC3K2(
                in_channels=base_channel * 2**4,
                out_channels=base_channel * 2**3,
                activation=activation,
                kernel_sizes=[kernel_size, kernel_size],
                num_c3k=3,
                expansion=expansion,
                shortcut=False,
            )

            self.downsample7 = PlanarCBNA(
                in_channels=base_channel * 2**3,
                out_channels=base_channel * 2**4,
                activation=activation,
                kernel_size=kernel_size,
                stride=2,
            )
            self.c3k2_downsample7 = PlanarC3K2(
                in_channels=base_channel * 2**5,
                out_channels=base_channel * 2**4,
                activation=activation,
                kernel_sizes=[kernel_size, kernel_size],
                num_c3k=3,
                expansion=expansion,
                shortcut=False,
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the full YOLOv11 backbone+neck.

        Args:
            x (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            dict[str, torch.Tensor]: Multi-scale feature maps keyed by stage name.
        """
        # Keeping internal copy because downstream layers outside backbone may need the features
        # Downsample
        down1 = self.downsample1(x)
        down2 = self.downsample2(down1)
        down3 = self.downsample3(down2)
        down4 = self.downsample4(down3)
        down5 = self.downsample5(down4)

        # Neck
        deep_features = self.neck(down5)

        # Upsample
        up1 = self.c3k2_upsample1(
            torch.cat([down4, self.upsample1(deep_features)], dim=1)
        )
        up2 = self.c3k2_upsample2(torch.cat([down3, self.upsample2(up1)], dim=1))

        output_dict = {"high_resolution": up2, "down1": down1, "down2": down2}

        if self.multilevel:
            # Multi-level outputs
            down6 = self.c3k2_downsample6(
                torch.cat([up1, self.downsample6(up2)], dim=1)
            )
            output_dict["mid_resolution"] = down6

            down7 = self.c3k2_downsample7(
                torch.cat([deep_features, self.downsample7(down6)], dim=1)
            )
            output_dict["low_resolution"] = down7

        return output_dict
