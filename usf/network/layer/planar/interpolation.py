#!/usr/bin/env python3
#
# Created on Fri Oct 31 2025 16:21:48
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarInterpolation(nn.Module):
    """Planar spatial interpolation layer wrapping ``F.interpolate``.

    Upsample or downsample a feature map by a given ``scale_factor`` using
    the specified interpolation ``mode``. Operates on NCHW tensors.
    """

    def __init__(
        self,
        scale_factor: float | None = None,
        mode: str | None = None,
        *args,
        **kwargs,
    ):
        """Modularized interpolation layer

        Args:
            scale_factor (float | None, optional): > 1 upsample, < 1 downsample. Defaults to None which is 1.
            mode (str | None, optional): interpolation function. Defaults to None which is bilinear interpolation.
        """
        super().__init__()
        self.scale_factor = scale_factor or 1.0
        self.mode = mode or "bilinear"
        self.args = args
        self.kwargs = kwargs

    def extra_repr(self) -> str:
        """Return a compact string summary for ``print(module)``.

        Returns:
            str: Scale factor, mode, and any extra args/kwargs.
        """
        extra_repr = f"scale_factor={self.scale_factor}, mode={self.mode}"

        if self.args:
            args_repr = ""
            for arg in self.args:
                args_repr += f", {arg}"
            extra_repr += f", args=({args_repr})"

        if self.kwargs:
            for key, value in self.kwargs.items():
                extra_repr += f", {key}={value}"

        return extra_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate the input tensor.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Interpolated feature map.
        """
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            *self.args,
            **self.kwargs,
        )
