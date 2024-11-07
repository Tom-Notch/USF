#!/usr/bin/env python3
#
# Created on Thu Mar 20 2025 16:11:17
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from typing import Any

import torch
import torch.nn as nn

from usf.network.layer.spherical.backend.weighting_function.weighting_function import (
    WeightingFunction,
)


class Discrete(WeightingFunction):
    """Discrete weighting function using learned embedding lookup for quantized input values."""

    def __init__(self, config: dict[str, Any], *args, **kwargs):
        """Initialize the Discrete weighting function

        Keys in config:
            value_min (float): min value
            value_max (float): max value
            num_slices (int): number of discretizations
        """
        super().__init__(config, *args, **kwargs)

        self.value_min = config["value_min"]
        self.value_max = config["value_max"]
        self.num_slices = config["num_slices"]

        assert (
            self.value_max > self.value_min
        ), f"Discretization value max must be larger than min, got value_max = {self.value_max} and value_min = {self.value_min}"
        assert (
            self.num_slices > 0
        ), f"Must have positive number of slices, got {self.num_slices}"

        assert (
            self.in_channels == 1
        ), f"in_channels must be 1 when using discrete weighting function, got {self.in_channels}"

        self.embedding = nn.Embedding(
            num_embeddings=self.num_slices,
            embedding_dim=self.num_functions * self.out_channels,
        )

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Slice count, value range, and number of functions.
        """
        return (
            f"num_slices={self.num_slices}"
            f", value_min={self.value_min}"
            f", value_max={self.value_max}"
            f", num_functions={self.num_functions}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): shape (..., WeightingFunction.in_channels)

        Returns:
            torch.Tensor: shape (..., num_functions, out_channels)
        """
        boundaries = torch.linspace(
            self.value_min,
            self.value_max,
            self.num_slices + 1,
            device=x.device,
            dtype=x.dtype,
        )[1:-1]

        bin_indices = torch.bucketize(x, boundaries=boundaries)

        return self.embedding(bin_indices).view(
            *x.shape[:-1], self.num_functions, self.out_channels
        )
