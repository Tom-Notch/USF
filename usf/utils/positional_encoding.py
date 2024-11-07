#!/usr/bin/env python3
#
# Created on Thu Aug 21 2025 01:28:39
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn


class Embedding(nn.Module, ABC):
    """Abstract base class for positional / frequency embeddings.

    Factory dispatch via ``__new__``: calling ``Embedding(config)`` returns the
    concrete subclass selected by ``config["type"]`` (one of ``"fourier"``,
    ``"cosine"``).

    Subclasses must implement:
        - ``out_channels`` (property): number of output channels produced by the
          embedding for a given ``in_channels``.
        - ``forward(x)`` -> ``torch.Tensor``: map input features to the embedded
          space.
    """

    _EMBEDDING_TYPES = {
        "fourier": (".positional_encoding", "FourierEmbedding"),
        "cosine": (".positional_encoding", "CosineEmbedding"),
    }

    def __new__(
        cls,
        config: dict[str, Any],
        *args,
        **kwargs,
    ) -> Embedding:
        """Create and return a concrete Embedding subclass based on ``config["type"]``.

        Args:
            config (dict[str, Any]): Must contain ``"type"`` (str) selecting the
                backend, plus any backend-specific keys.

        Returns:
            Embedding: An instance of the selected concrete subclass.
        """
        embedding_type = config.get("type", None)
        assert embedding_type is not None, "Key 'type' must be provided in config dict"

        embedding_class_config = cls._EMBEDDING_TYPES.get(embedding_type, None)
        assert (
            embedding_class_config is not None
        ), f"unsupported embedding type {embedding_type}, supported types: {cls._EMBEDDING_TYPES.keys()}"

        module_path, class_name = embedding_class_config

        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        subclass_instance.__init__(config, *args, **kwargs)
        return subclass_instance

    def __init__(
        self,
        config: dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        """Store shared config and derive ``in_channels``.

        Args:
            config (dict[str, Any]): Embedding configuration dictionary.
                Must contain ``"in_channels"`` (int).
        """
        super().__init__()
        self.config = config
        self.in_channels = config["in_channels"]

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of output channels produced by this embedding.

        Returns:
            int: Dimensionality of the embedded output.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input features to the embedded space.

        Args:
            x (torch.Tensor): Input of shape (..., in_channels).

        Returns:
            torch.Tensor: Embedded output of shape (..., out_channels).
        """
        raise NotImplementedError


class FourierEmbedding(Embedding):
    """Fourier embedding with sin+cos at exponentially spaced frequencies.

    Output per input channel:
    ``[sin(2^0·b·x), cos(2^0·b·x), ..., sin(2^(L-1)·b·x), cos(2^(L-1)·b·x)]``
    optionally prepended by the raw ``x`` value.
    """

    def __init__(self, config: dict[str, Any], *args, **kwargs) -> None:
        """Initialize FourierEmbedding.

        Args:
            config (dict[str, Any]): Must contain:
                - ``in_channels`` (int): Number of input channels.
                - ``L`` (int, optional): Number of Fourier levels. Defaults to 3.
                - ``basis`` (float, optional): Base frequency multiplier. Defaults to 1.0.
                - ``include_x`` (bool, optional): Prepend raw input. Defaults to True.
        """
        super().__init__(config, *args, **kwargs)

        self.L = config.get("L", 3)
        self.basis = config.get("basis", 1.0)
        self.include_x = config.get("include_x", True)

        assert (
            self.L >= 0
        ), f"Number of fourier levels L must be non-negative, got {self.L}"
        assert self.basis > 0, f"Fourier basis must be positive, got {self.basis}"

        self.register_buffer(
            "factors",
            (2 ** torch.arange(self.L)) * self.basis,
        )  # (L,)

    def extra_repr(self) -> str:
        """Return a string summary for ``print(module)``.

        Returns:
            str: Compact representation of key hyperparameters.
        """
        return (
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", num_fourier_levels={self.L}"
            f", basis={self.basis:.3f}"
            f", include_x={self.include_x}"
        )

    @property
    def out_channels(self) -> int:
        """Number of output channels: ``in_channels * ((1 if include_x) + 2 * L)``.

        Returns:
            int: Output dimensionality.
        """
        return self.in_channels * (int(self.include_x) + 2 * self.L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to Fourier features: ``[(x, )sin(2^0·b·x), ..., cos(2^(L-1)·b·x)]``.

        Args:
            x (torch.Tensor): input of shape (..., D)

        Returns:
            torch.Tensor: shape (..., D x ((1 + )2 x L))
        """
        assert (
            x.shape[-1] == self.in_channels
        ), f"Input shape must match in_channels, input has {x.shape[-1]} channels, expect {self.in_channels} in_channels"

        if self.L == 0:
            return x if self.include_x else x.new_empty((*x.shape[:-1], 0))

        x = x.unsqueeze(-1)  # shape (..., D, 1)

        embed_list = [x] if self.include_x else []  # shape (..., D, 1)

        # Multiply x (shape: (..., D)) by factors via broadcasting -> (..., D, L)
        x_expanded = x * self.factors  # shape: (..., D, L)

        embed_list.extend([torch.sin(x_expanded), torch.cos(x_expanded)])

        embed = torch.cat(embed_list, dim=-1)  # shape (..., D, (1 + )2 x L)

        return embed.flatten(start_dim=-2)  # shape (..., D x ((1 + )2 x L))


class CosineEmbedding(Embedding):
    """Cosine-only embedding with exponentially spaced frequencies.

    Basis: [cos(2⁰·k₀·x), cos(2¹·k₀·x), ..., cos(2^(L-1)·k₀·x)]

    The base frequency k₀ is typically computed by the spherical layer as floor(π / r)
    where r is the kernel radius, ensuring the first mode does not have a mirror axis
    inside the support [0, r].

    This embedding is even (f(x) = f(-x)) and 2π-periodic (since all 2^l·k₀ are
    integers when k₀ is an integer), which is the correct structure for geodesic
    distance weighting on the sphere.

    Keys in config:
        in_channels (int): number of input channels.
        L (int): number of cosine harmonics (must be > 0).
        basis (float): base frequency k₀. Higher harmonics are exponential multiples.
    """

    def __init__(self, config: dict[str, Any], *args, **kwargs) -> None:
        """Initialize CosineEmbedding.

        Args:
            config (dict[str, Any]): Must contain:
                - ``in_channels`` (int): Number of input channels.
                - ``L`` (int): Number of cosine harmonics (must be > 0).
                - ``basis`` (float): Base frequency k₀.
        """
        super().__init__(config, *args, **kwargs)

        self.L = config["L"]
        self.basis = config["basis"]

        assert self.L > 0, f"Number of harmonics L must be positive, got {self.L}"
        assert self.basis > 0, f"basis must be positive, got {self.basis}"

        self.register_buffer(
            "factors",
            (2 ** torch.arange(self.L)) * self.basis,
        )  # (L,)

    def extra_repr(self) -> str:
        """Return a string summary for ``print(module)``.

        Returns:
            str: Compact representation of key hyperparameters.
        """
        return (
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", L={self.L}"
            f", basis={self.basis:.3f}"
        )

    @property
    def out_channels(self) -> int:
        """Number of output channels: ``in_channels * L``.

        Returns:
            int: Output dimensionality.
        """
        return self.in_channels * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps x to [cos(2⁰·k₀·x), cos(2¹·k₀·x), ..., cos(2^(L-1)·k₀·x)] per input channel.

        Args:
            x (torch.Tensor): input of shape (..., D)

        Returns:
            torch.Tensor: shape (..., D * L)
        """
        assert (
            x.shape[-1] == self.in_channels
        ), f"Input shape must match in_channels, input has {x.shape[-1]} channels, expect {self.in_channels} in_channels"

        x = x.unsqueeze(-1)  # (..., D, 1)
        embedded = torch.cos(x * self.factors)  # (..., D, L)
        return embedded.flatten(start_dim=-2)  # (..., D * L)
