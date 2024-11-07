#!/usr/bin/env python3
#
# Created on Mon Mar 03 2025 13:37:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn

from usf.network.layer.attention import Attention


class PlanarSelfAttention(nn.Module):
    """Multi-head self-attention layer for planar (image-grid) feature maps.

    Flattens spatial dimensions to a token sequence, applies layer-norm +
    multi-head attention (no positional encoding), and adds a residual
    connection. The attention backend is selected via the ``backend`` argument.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        backend: str | None = None,
        pre_normalize_value: bool = False,
    ):
        """Initialize PlanarSelfAttention.

        Args:
            in_channels (int): Number of input/output channels.
            num_heads (int): Number of attention heads.
            backend (str | None, optional): backend to use, can choose from
                {"xformers", "flex attention", "plain"}. Defaults to None.
            pre_normalize_value (bool, optional): whether to apply layer norm
                to value vectors before self-attention. Defaults to False.
        """
        super().__init__()

        assert (
            in_channels % num_heads == 0
        ), f"in_channels ({in_channels}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.embedding_dim = in_channels // self.num_heads

        self.backend = backend or "xformers"
        self.attention = Attention(backend=self.backend)

        self.pre_normalize_value = pre_normalize_value

        self.layer_norm = nn.LayerNorm(normalized_shape=in_channels)

        self.query_projection = nn.Linear(
            in_features=in_channels,
            out_features=in_channels,
            bias=False,
        )
        self.key_projection = nn.Linear(
            in_features=in_channels,
            out_features=in_channels,
            bias=False,
        )
        self.value_projection = nn.Linear(
            in_features=in_channels,
            out_features=in_channels,
            bias=False,
        )

        self.output_projection = nn.Linear(
            in_features=in_channels,
            out_features=in_channels,
            bias=True,
        )

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Channel count, head count, and key feature flags.
        """
        return (
            f"{self.num_heads * self.embedding_dim}"
            + f", num_heads={self.num_heads}"
            + f", embedding_dim={self.embedding_dim}"
            + f", pre_normalize_value={self.pre_normalize_value}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        B, C, H, W = x.shape
        N = H * W

        # Flatten spatial dimensions for attention
        x_flattened = (
            x.clone().view(B, C, N).transpose(2, 1).contiguous()
        )  # Shape: (B, N, C)
        x_normalized = self.layer_norm(x_flattened)

        value_input = x_normalized if self.pre_normalize_value else x_flattened

        query = (
            self.query_projection(x_normalized)
            .view(B, N, self.num_heads, self.embedding_dim)
            .contiguous()
        )  # (B, N, H, D)
        key = (
            self.key_projection(x_normalized)
            .view(B, N, self.num_heads, self.embedding_dim)
            .contiguous()
        )  # (B, N, H, D)
        value = (
            self.value_projection(value_input)
            .view(B, N, self.num_heads, self.embedding_dim)
            .contiguous()
        )  # (B, N, H, D)

        attended_value = self.attention(
            query=query,
            key=key,
            value=value,
            attention_bias=None,
        )  # (B, N, H, D)

        output_value = x + self.output_projection(
            attended_value.reshape(B, N, C)
        ).transpose(2, 1).reshape(
            B, C, H, W
        )  # standard residual update

        return output_value
