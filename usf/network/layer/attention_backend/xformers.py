#!/usr/bin/env python3
#
# Created on Wed Oct 29 2025 21:46:56
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

from math import ceil
from typing import Any

import torch
from xformers import ops as xops

from usf.network.layer.attention import Attention


class XFormers(Attention):
    """xFormers memory-efficient attention backend."""

    def __init__(self, *, backend: str = "xformers") -> None:
        """Initialize xFormers backend.

        Args:
            backend (str): Must be ``"xformers"``. Used for factory dispatch.
        """
        assert (
            backend == "xformers"
        ), f"wrong backend instantiated, got backend = {backend}"
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor | Any | None,
    ) -> torch.Tensor:
        """Attention implementation with xFormers kernel xops.memory_efficient_attention

        Args:
            query (torch.Tensor): shape (B, N_q, H, D)
            key (torch.Tensor): shape (B, N_kv, H, D)
            value (torch.Tensor): shape (B, N_kv, H, D_v)
            attention_bias (torch.Tensor | Any | None): shape (B, H, N_q, N_kv)

        Returns:
            torch.Tensor: shape (B, N_q, H, D_v)
        """
        if isinstance(attention_bias, torch.Tensor):
            # Pad last dim to x 8
            B, H, N_q, N_kv = attention_bias.shape
            padded_bias = torch.empty(
                (B, H, N_q, ceil(N_kv / 8) * 8),
                dtype=attention_bias.dtype,
                device=attention_bias.device,
            )
            padded_bias[..., :N_kv].copy_(attention_bias)
            attention_bias = padded_bias[..., :N_kv]

        return xops.memory_efficient_attention(
            query=query,
            key=key,
            value=value,
            attn_bias=attention_bias,
        )
