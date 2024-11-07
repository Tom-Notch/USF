#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 00:52:14
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.layer.planar.interpolation import PlanarInterpolation
from usf.utils.torch_numpy import autopad


class PlanarCBNA(nn.Sequential):
    """
    Planar Convolution + Batch Normalization + Activation.
    Supports both standard convolutions and transposed convolutions (upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | bool | nn.Module | None = None,
        activation_config: dict = {"inplace": True},
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int | None = None,  # Auto-computed with autopad
        dilation: int = 1,
        upsample: bool | str | None = None,
        *args,
        **kwargs,
    ):
        """Initialize PlanarCBNA.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str | bool | nn.Module | None, optional): Activation function name, module, or False/None to disable.
            activation_config (dict, optional): Kwargs passed to activation constructor. Defaults to ``{"inplace": True}``.
            kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride. Defaults to 1.
            padding (int | None, optional): Padding (auto-computed if None).
            dilation (int, optional): Dilation rate. Defaults to 1.
            upsample (bool | str | None, optional): If True or ``"interpolation"``, use interpolation + 1x1 Conv; if ``"conv"``, use ConvTranspose2d. Defaults to None (standard Conv2d).
        """
        padding = autopad(kernel_size, padding, dilation)

        layer_list: list[tuple[str, nn.Module]] = []

        if upsample is None or upsample is False:
            layer_list.append(
                (
                    "conv",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                        *args,
                        **kwargs,
                    ),
                )
            )
        elif upsample == "interpolation" or upsample is True:
            layer_list.append(
                ("upsample", PlanarInterpolation(scale_factor=stride, mode="bilinear"))
            )
            layer_list.append(
                (
                    "conv1x1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        *args,
                        **kwargs,
                    ),
                )
            )
        elif upsample == "conv":
            layer_list.append(
                (
                    "transpose_conv",
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        output_padding=1 if stride > 1 else 0,
                        bias=False,
                        *args,
                        **kwargs,
                    ),
                )
            )
        else:
            raise ValueError(f"upsample arg {upsample} unimplemented")

        layer_list.append(("batchnorm", nn.BatchNorm2d(out_channels)))

        # Handle activation logic
        if activation is None or activation is False:
            pass
        elif isinstance(activation, str):
            layer_list.append(
                ("activation", getattr(nn, activation)(**activation_config))
            )  # Dynamically instantiate activation
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            layer_list.append(("activation", activation(**activation_config)))
        elif isinstance(activation, nn.Module):  # already instantiated
            layer_list.append(("activation", activation))
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

        # Define the sequential module
        super().__init__(OrderedDict(layer_list))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Conv → BN → Activation sequentially, ensuring contiguous memory.

        Args:
            input (torch.Tensor): Input of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output of shape (B, C_out, H', W').
        """
        return super().forward(input.contiguous())
