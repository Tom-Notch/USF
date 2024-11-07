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
from matplotlib import pyplot as plt

from usf.network.layer.attention import Attention


class Plain(Attention):
    """Plain PyTorch scaled-dot-product attention (SDPA) backend with optional visualization."""

    def __init__(self, *, backend: str = "plain") -> None:
        """Initialize plain attention backend.

        Args:
            backend (str): Must be ``"plain"``. Used for factory dispatch.
        """
        assert (
            backend == "plain"
        ), f"wrong backend instantiated, got backend = {backend}"
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor | Any | None,
    ) -> torch.Tensor:
        """Attention implementation with plain PyTorch operations

        Args:
            query (torch.Tensor): shape (B, N_q, H, D)
            key (torch.Tensor): shape (B, N_kv, H, D)
            value (torch.Tensor): shape (B, N_kv, H, D_v)
            attention_bias (torch.Tensor | Any | None): shape (B, H, N_q, N_kv)

        Touches:
            self.raw_attention_scores (torch.Tensor): attention score before adding bias
            self.biased_attention_scores (torch.Tensor): attention score after adding bias
            self.bias (torch.Tensor): attention bias

        Returns:
            torch.Tensor: shape (B, N_q, H, D_v)
        """
        B, N_q, H, D = query.shape
        N_kv = key.shape[1]
        D_v = value.shape[-1]

        query = query * (D**-0.5)  # scale

        # (B, H, N_q, D) x (B, H, D, N_kv) -> (B, H, N_q, N_kv)
        query_bh = query.permute(0, 2, 1, 3).reshape(B * H, N_q, D)  # (B x H, N_q, D)
        key_bh = key.permute(0, 2, 3, 1).reshape(B * H, D, N_kv)  # (B x H, D, N_kv)
        scores = (query_bh @ key_bh).reshape(B, H, N_q, N_kv)  # (B, H, N_q, N_kv)

        self.raw_attention_scores = scores.clone()

        if attention_bias is not None:
            # bias already in (B, H, N_q, N_kv)
            scores = scores + attention_bias
            self.attention_bias = attention_bias.clone()
            self.biased_attention_scores = scores.clone()
        else:
            if hasattr(self, "attention_bias"):
                del self.attention_bias
            if hasattr(self, "biased_attention_scores"):
                del self.biased_attention_scores

        # Softmax on key dimension
        probabilities = scores.softmax(dim=-1)  # (B, H, N_q, N_kv)

        # Context: (B, H, N_q, N_kv) x (B, H, N_kv, D_v) -> (B, H, N_q, D_v)
        value_bh = value.permute(0, 2, 1, 3).reshape(
            B * H, N_kv, D_v
        )  # (B x H, N_kv, D_v)
        attended_value = (
            probabilities.reshape(B * H, N_q, N_kv) @ value_bh
        )  # (B x H, N_q, D_v)
        attended_value = attended_value.reshape(B, H, N_q, D_v).permute(
            0, 2, 1, 3
        )  # (B, N_q, H, D_v)

        return attended_value

    @staticmethod
    def _tensor_minmax(tensor: torch.Tensor) -> tuple[float, float]:
        """Return (min, max) ignoring NaN/Inf, or None if tensor is None."""
        flat = tensor.flatten()
        mask = torch.isfinite(flat)
        if not torch.any(mask):
            return (0.0, 0.0)
        valid = flat[mask]
        return valid.min().item(), valid.max().item()

    @staticmethod
    def _plot_heatmap(
        matrix: torch.Tensor,
        title: str,
        value_range: tuple[float, float] | None,
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(
            matrix,
            aspect="auto",
            vmin=value_range[0],
            vmax=value_range[1],
        )
        ax.set_xlabel("Key index (Nkv)")
        ax.set_ylabel("Query index (Nq)")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("value range", rotation=90, va="center")
        fig.tight_layout()
        return fig

    def visualize_attention(
        self,
        num_visualize_samples: int | None = None,
        num_visualize_heads: int | None = None,
    ) -> dict[str, list[list[plt.Figure]]]:
        """Visualize raw attention scores, show bias and biased scores if available

        Args:
            num_visualize_samples (int | None, optional): number of visualized samples in a batch
            num_visualize_heads (int | None, optional): number of heads visualized in a sample

        Reads:
            self.raw_attention_scores (torch.Tensor): attention score before adding bias
            self.biased_attention_scores (torch.Tensor): attention score after adding bias
            self.bias (torch.Tensor): attention bias

        Returns:
            dict[str, list[list[plt.Figure]]]: indexed by type first ("raw score", "bias", "biased score"), then double list, first layer is batch, second is head
        """
        if not hasattr(self, "raw_attention_scores"):
            raise RuntimeError("No attention snapshot. Run a forward() first.")

        # shape: (B, H, N_q, N_kv)
        raw = self.raw_attention_scores.detach().float().cpu()

        has_bias = hasattr(self, "attention_bias")
        has_biased = hasattr(self, "biased_attention_scores")

        bias = self.attention_bias.detach().float().cpu() if has_bias else None
        biased = (
            self.biased_attention_scores.detach().float().cpu() if has_biased else None
        )

        B, H, N_q, N_kv = raw.shape
        num_visualize_samples = max(1, min(B, num_visualize_samples or B))
        num_visualize_heads = max(1, min(H, num_visualize_heads or H))

        visualizations_dict: dict[str, list[list[plt.Figure]]] = {}

        for b in range(num_visualize_samples):
            raw_range = self._tensor_minmax(raw[b])
            bias_range = self._tensor_minmax(bias[b]) if has_bias else None
            biased_range = self._tensor_minmax(biased[b]) if has_biased else None

            raw_row, bias_row, biased_row = [], [], []

            for h in range(num_visualize_heads):
                # raw
                raw_row.append(
                    self._plot_heatmap(
                        raw[b, h],
                        f"Raw Score | Sample {b + 1}, Head {h + 1} | ({N_q}, {N_kv})",
                        raw_range,
                    )
                )

                # bias (optional)
                if has_bias and bias is not None:
                    bias_row.append(
                        self._plot_heatmap(
                            bias[b, h],
                            f"Bias | Sample {b + 1}, Head {h + 1} | ({N_q}, {N_kv})",
                            bias_range,
                        )
                    )

                # biased scores (optional)
                if has_biased and biased is not None:
                    biased_row.append(
                        self._plot_heatmap(
                            biased[b, h],
                            f"Biased Score | Sample {b + 1}, Head {h + 1} | ({N_q}, {N_kv})",
                            biased_range,
                        )
                    )

            visualizations_dict.get("raw score", []).append(raw_row)
            if has_bias:
                visualizations_dict.get("bias", []).append(bias_row)
            if has_biased:
                visualizations_dict.get("biased score", []).append(biased_row)

        return visualizations_dict
