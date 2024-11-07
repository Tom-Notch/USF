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

from typing import Any

import torch
from torch.nn.attention.flex_attention import flex_attention

from usf.network.layer.attention import Attention


class FlexAttention(Attention):
    """PyTorch flex_attention backend with compile-friendly score modification."""

    def __init__(self, *, backend: str = "flex attention") -> None:
        """Initialize flex_attention backend.

        Args:
            backend (str): Must be ``"flex attention"``. Used for factory dispatch.
        """
        assert (
            backend == "flex attention"
        ), f"wrong backend instantiated, got backend = {backend}"
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor | Any | None,
    ) -> torch.Tensor:
        """Attention implementation with PyTorch kernel torch.nn.attention.flex_attention.flex_attention

        Args:
            query (torch.Tensor): shape (B, N_q, H, D)
            key (torch.Tensor): shape (B, N_kv, H, D)
            value (torch.Tensor): shape (B, N_kv, H, D_v)
            attention_bias (torch.Tensor | Any | None): shape (B, H, N_q, N_kv)

        Returns:
            torch.Tensor: shape (B, N_q, H, D_v)
        """

        if isinstance(attention_bias, torch.Tensor):

            def score_mod(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                k_idx: torch.Tensor,
            ) -> torch.Tensor:
                """Add pre-computed attention bias to the raw attention score."""
                return score + attention_bias[batch, head, q_idx, k_idx]

        else:
            score_mod = None

        return (
            flex_attention(
                query=query.transpose(-2, -3).contiguous(),
                key=key.transpose(-2, -3).contiguous(),
                value=value.transpose(-2, -3).contiguous(),
                score_mod=score_mod,
            )
            .transpose(-2, -3)
            .contiguous()
        )
