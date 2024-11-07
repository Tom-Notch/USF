#!/usr/bin/env python3
#
# Created on Sun Feb 16 2025 20:26:40
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch.nn as nn

from usf.network.layer.attention import Attention
from usf.utils.spherical_image import BatchSphericalImage


class SphericalSelfAttention(nn.Module):
    """Multi-head self-attention layer for spherical signals.

    Operates on ``BatchSphericalImage`` token sequences with vanilla
    scaled-dot-product attention (no positional encoding). The attention
    backend (``xformers``, ``flex_attention``, or ``plain``) is selected
    via the ``backend`` argument.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        backend: str | None = None,
        pre_normalize_value: bool = False,
    ):
        """Initialize SphericalSelfAttention.

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

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass of self-attention.

        Args:
            batch_spherical_image (BatchSphericalImage): Input spherical image.

        Returns:
            BatchSphericalImage: Output with the same vector but updated
                batch_value in the same shape.
        """
        # shape: (B, N, C)
        batch_value = batch_spherical_image.batch_value
        normalized_batch_value = self.layer_norm(batch_value)
        B, N, C = normalized_batch_value.shape

        value_input = (
            normalized_batch_value if self.pre_normalize_value else batch_value
        )

        query = (
            self.query_projection(normalized_batch_value)
            .view(B, N, self.num_heads, self.embedding_dim)
            .contiguous()
        )  # (B, N, H, D)
        key = (
            self.key_projection(normalized_batch_value)
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

        output_batch_value = batch_value + self.output_projection(
            attended_value.reshape(B, N, C)
        )  # standard residual update

        return BatchSphericalImage(
            batch_value=output_batch_value,
            vector=batch_spherical_image.vector,
        )
