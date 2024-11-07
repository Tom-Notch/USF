#!/usr/bin/env python3
#
# Created on Thu Mar 20 2025 16:11:00
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from opt_einsum import contract

from usf.network.layer.spherical.backend.weighting_function.weighting_function import (
    WeightingFunction,
)
from usf.utils.positional_encoding import Embedding


class Linear(nn.Module):
    """
    A linear layer applied independently over a number of functions.
    For an input of shape (..., num_functions, in_channels), it produces an output
    of shape (..., num_functions, out_channels) using a separate weight matrix per function.
    """

    def __init__(
        self,
        num_functions: int,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        """Initialize per-function linear layer.

        Args:
            num_functions (int): Number of independent linear transforms.
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            bias (bool, optional): Include bias. Defaults to True.
        """
        super().__init__()

        self.num_functions = num_functions
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a weight tensor for each function: shape (num_functions, in_channels, out_channels)
        self.weight = nn.Parameter(
            torch.Tensor(num_functions, self.in_channels, self.out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_functions, self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> Linear:
        """Reinitialize weights (Kaiming uniform) and biases (uniform).

        Returns:
            Linear: ``self`` for chaining.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        return self

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: in/out channels and bias flag.
        """
        return (
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", bias={getattr(self, 'bias', None) is not None}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-function linear transform.

        Args:
            x (torch.Tensor): shape (..., num_functions, in_channels).

        Returns:
            torch.Tensor: shape (..., num_functions, out_channels).
        """
        # x shape: (..., num_functions, in_channels)
        batch_shape = x.shape[:-2]

        # Each function is applied separately
        out = contract(
            "bni,nio->bno",
            x.view(-1, x.shape[-2], x.shape[-1]),
            self.weight,
            memory_limit="max_input",
        )

        # # (B, num_functions, 1, in_channels) @ (num_functions, in_channels, out_channels)
        # # = (B, num_functions, 1, out_channels)
        # out = (x.view(-1, x.shape[-2], 1, x.shape[-1]) @ self.weight).squeeze(-2)

        # out shape: (B, num_functions, out_channels)
        if self.bias is not None:
            out += self.bias.unsqueeze(0)

        # out shape: (..., num_functions, out_channels)
        return out.view(*batch_shape, out.shape[-2], out.shape[-1])


class MLP(nn.Module):
    """
    A stack of independent MLPs. Each of the num_functions independent functions processes
    the same input (after replication) but with its own set of weights.
    """

    def __init__(
        self,
        num_functions: int,
        in_channels: int,
        out_channels: int,
        hidden_dims: list[int],
        activation: str,
    ) -> None:
        """Initialize per-function MLP.

        Args:
            num_functions (int): Number of independent MLP heads.
            in_channels (int): Input feature dimension.
            out_channels (int): Output feature dimension.
            hidden_dims (list[int]): Hidden layer widths.
            activation (str): Activation function name (must exist in ``torch.nn``).
        """
        super().__init__()

        self.num_functions = num_functions

        if getattr(nn, activation, None) is not None:
            act_class = getattr(nn, activation)
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        dims = [in_channels] + hidden_dims + [out_channels]
        for i in range(len(dims) - 1):
            layers[f"Linear{i+1}"] = Linear(
                self.num_functions,
                dims[i],
                dims[i + 1],
                bias=True,
            )
            layers[f"Activation{i+1}"] = act_class(inplace=True)
        del layers[f"Activation{len(dims) - 1}"]  # remove last activation

        self.model = nn.Sequential(layers)

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Number of independent functions.
        """
        return f"num_functions={self.num_functions}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): shape (..., in_channels)

        Returns:
            torch.Tensor: shape (..., num_functions, out_channels)
        """
        # x shape: (..., in_channels)
        batch_shape = x.shape[:-1]

        # Replicate the input for each independent function, x_expand shape: (..., num_functions, in_channels)
        x_expand = x.unsqueeze(-2).expand(*batch_shape, self.num_functions, x.shape[-1])

        # shape: (..., num_functions, out_channels)
        return self.model(x_expand)


class Continuous(WeightingFunction):
    """Continuous weighting function: positional embedding followed by per-function MLP."""

    def __init__(self, config: dict[str, Any], *args, **kwargs):
        """Initialize the Continuous weighting function

        Keys in config:
            embedding (dict): embedding config with "type" key and type-specific params.
                e.g. {"type": "cosine", "in_channels": 1, "L": 6, "basis": 15.0}
                e.g. {"type": "fourier", "in_channels": 1, "L": 8, "basis": 3.14159, "include_x": true}
            hidden_dims (list[int]): hidden dims for MLP
            activation (str): activation for MLP
        """
        super().__init__(config, *args, **kwargs)

        assert all(
            dim > 0 for dim in config["hidden_dims"]
        ), "hidden_dims must all be positive"

        embedding_config = config["embedding"].copy()
        embedding_config["in_channels"] = self.in_channels
        self.embedder = Embedding(embedding_config)

        self.MLP = MLP(
            in_channels=self.embedder.out_channels,
            out_channels=self.out_channels,
            num_functions=self.num_functions,
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): shape (..., WeightingFunction.in_channels)

        Returns:
            torch.Tensor: shape (..., num_functions, out_channels)
        """
        return self.MLP(self.embedder(x))
