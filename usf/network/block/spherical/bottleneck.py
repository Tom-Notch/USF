#!/usr/bin/env python3
#
# Created on Tue Feb 18 2025 00:59:04
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.utils.spherical_image import BatchSphericalImage


class SphericalBottleNeck(nn.Module):
    """Spherical bottleneck block with two sequential CBNA layers and optional residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        radius: float,
        weighting_function_config: dict,
        shortcut: bool = True,
        expansion: float = 0.5,
        *args,
        **kwargs,
    ):
        """Initialize the spherical bottleneck block.

        Args:
            in_channels (int): Input channel dimension.
            out_channels (int): Output channel dimension.
            activation (str): Activation function name.
            radius (float): Geodesic radius of the first and second conv.
            weighting_function_config (dict): Weighting function config for the spherical convs.
            shortcut (bool, optional): Whether to use residual shortcut. Defaults to True.
            expansion (float, optional): Hidden channels = out_channels x expansion. Defaults to 0.5.
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
                        SphericalCBNA(
                            in_channels=in_channels,
                            out_channels=hidden_channels,
                            weighting_function_config=weighting_function_config,
                            radius=radius,
                            activation=activation,
                            identical_output_vector=True,
                            *args,
                            **kwargs,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=hidden_channels,
                            out_channels=out_channels,
                            weighting_function_config=weighting_function_config,
                            radius=radius,
                            activation=activation,
                            identical_output_vector=True,
                            *args,
                            **kwargs,
                        ),
                    ),
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass through the spherical bottleneck with optional residual connection.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Output with updated ``batch_value``.
        """
        conv_output = self.cbna_block(batch_spherical_image)

        batch_value = (
            batch_spherical_image.batch_value + conv_output.batch_value
            if self.shortcut
            else conv_output.batch_value
        )

        return BatchSphericalImage(
            batch_value=batch_value,
            vector=batch_spherical_image.vector,
            mask=batch_spherical_image.mask,
        )
