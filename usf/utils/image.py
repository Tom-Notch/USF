#!/usr/bin/env python3
#
# Created on Thu May 29 2025 23:14:30
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import io

import numpy as np
from matplotlib import pyplot as plt


def fig_to_numpy(
    fig: plt.Figure,
    *,
    dpi: int = 100,
) -> np.ndarray:
    """
    Convert a matplotlib figure to a NumPy array.

    Args:
        fig (plt.Figure): The matplotlib figure to convert.
        dpi (int, optional): Dots per inch. Defaults to 100.

    Returns:
        np.ndarray: The figure as a NumPy array.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
        transparent=True,
    )
    buffer.seek(0)
    image = plt.imread(buffer)  # (H, W, 4) RGBA float32 [0, 1]
    image = (image * 255).astype(np.uint8)  # convert to uint8
    buffer.close()
    return image


def fig_to_pdf(
    fig: plt.Figure,
    path: str,
    *,
    dpi: int = 300,
    transparent: bool = True,
) -> None:
    """
    Save a matplotlib figure as a high-quality PDF file.

    Args:
        fig (plt.Figure): The matplotlib figure to save.
        path (str): Output PDF file path.
        dpi (int, optional): Dots per inch for rasterized elements. Defaults to 300.
        transparent (bool, optional): Whether to make the background transparent.
            Defaults to True.
    """
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi,
        transparent=transparent,
    )
