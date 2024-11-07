#!/usr/bin/env python3
#
# Created on Fri Jan 31 2025 09:13:18
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import matplotlib.cm as cm
import numpy as np
import torch

from usf.utils.spherical import cartesian2polar
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import to_numpy, to_torch


def visualize_spherical_channels(
    batch_spherical_image: BatchSphericalImage,
    binary: bool = False,
    colormap: str = "plasma",
    eps: float = 1e-8,
    value_range: tuple[float | None, float | None] = (None, None),
) -> list[BatchSphericalImage]:
    """
    Convert the batch_spherical_image.batch_value (B, N, C) into a list of B BatchSphericalImage,
    where each BatchSphericalImage's batch_value is rearranged to (C, N, 3) for visualization.
    For each channel, values are normalized using a specified value range:
      - If value_range is (None, None), the global per-channel minimum and maximum are computed across the entire batch.
      - Otherwise, the provided (min, max) values are used for normalization.
    A matplotlib colormap is then applied to each normalized channel to produce an RGB image with values in [0, 255].

    When binary=True, nonzero values are set to 1 (zeros remain 0) before applying the colormap.

    Args:
        batch_spherical_image (BatchSphericalImage):
            - .batch_value: Tensor of shape (B, N, C), typically float.
            - .vector: Geometry (kept unchanged).
            - .polar: Geometry (kept unchanged).
        binary (bool): If True, set nonzero values to 1 and zeros to 0.
        colormap (str): Name of the matplotlib colormap to use.
        eps (float): Small constant to avoid division by zero.
        value_range (tuple[float | None, float | None]): A tuple (min, max) to use for normalization.
            If both are None, the global min and max are computed from the batch.

    Returns:
        list[BatchSphericalImage]: A list of length B. For each sample, the BatchSphericalImage has a
        batch_value of shape (C, N, 3) where each channel is visualized as an RGB image with dtype uint8.
    """
    # Get input dimensions: (B, N, C)
    B, N, C = batch_spherical_image.batch_value.shape
    device = batch_spherical_image.batch_value.device
    vector = batch_spherical_image.vector

    # Reshape to (B*N, C) to compute global min and max for each channel if needed.
    reshaped = batch_spherical_image.batch_value.reshape(-1, C)  # (B*N, C)
    if value_range[0] is None:
        channel_min = torch.min(reshaped, dim=0)[0]  # (C,)
    else:
        channel_min = torch.full(
            (C,),
            value_range[0],
            dtype=batch_spherical_image.batch_value.dtype,
            device=device,
        )

    if value_range[1] is None:
        channel_max = torch.max(reshaped, dim=0)[0]  # (C,)
    else:
        channel_max = torch.full(
            (C,),
            value_range[1],
            dtype=batch_spherical_image.batch_value.dtype,
            device=device,
        )

    # Get the colormap (returns a function mapping [0,1] to RGBA)
    cmap = cm.get_cmap(colormap)

    batch_spherical_image_list = []
    # Process each sample individually.
    for b in range(B):
        channel_images = []
        # Process each channel.
        for c in range(C):
            values = batch_spherical_image.batch_value[b, :, c]  # shape (N,)
            if binary:
                normalized = (values != 0).to(torch.float32)  # binary: 0 or 1
            else:
                value_min = channel_min[c]
                value_max = channel_max[c]
                normalized = (values - value_min) / (value_max - value_min + eps)
                normalized = torch.clamp(normalized, 0.0, 1.0)
            # Apply colormap; drop alpha channel. Returned shape (N, 3) scaled to [0, 255]
            colors_np = cmap(to_numpy(normalized))[:, :3] * 255
            channel_images.append(
                SphericalImage(
                    value=to_torch(colors_np, device=device, dtype=torch.int16),
                    vector=vector,
                )
            )
        batch_spherical_image_list.append(
            BatchSphericalImage(batch_value=channel_images)
        )

    return batch_spherical_image_list


def sample_circles(
    center_vector: np.ndarray,
    r: np.ndarray,
    circle_spacing: float = 2 * np.pi / 5000,
    min_samples: int = 3,
) -> np.ndarray:
    """Sample points centered at center_vector on a unit sphere with geodesic radius r, each ring has same density determined by circle_spacing.

    Args:
        center_vector (np.ndarray): np array of shape (N, 3), each row is a center to sample rings
        r (np.ndarray): np array of shape (num_rings), indicates the geodesic radius of rings at a single center
        circle_spacing (float): the spacing between sampled points on each ring, default is 2 * pi / 5000
        min_samples (int): the minimum number of samples for each ring, default is 3

    Returns:
        np.ndarray: np array of shape (N, num_rings x num_sample, 3) for better colorization
                    color of each ring should match its center
    """
    # List to collect the ring templates for each ring radius.
    ring_templates_list = []

    # Define the canonical basis vectors.
    ex = np.array([1, 0, 0], dtype=np.float32)
    ey = np.array([0, 1, 0], dtype=np.float32)
    ez = np.array([0, 0, 1], dtype=np.float32)

    # Loop over each ring radius (acceptable since num_rings is small)
    for r_i in r:
        r_i = np.float32(r_i)
        # Compute the number of sample points for this ring.
        n_samples = int(np.ceil(2 * np.pi * np.sin(r_i) / circle_spacing))
        n_samples = max(n_samples, min_samples)

        # Generate sample angles for this ring.
        alpha = np.linspace(
            0, 2 * np.pi, n_samples, endpoint=False, dtype=np.float32
        )  # shape: (n_samples,)

        # Compute the ring points in the canonical configuration.
        # p(α) = cos(r_i) * e_x + sin(r_i) * (cos(α)*e_y + sin(α)*e_z)
        p = np.cos(r_i) * ex + np.sin(r_i) * (
            np.cos(alpha)[:, None] * ey + np.sin(alpha)[:, None] * ez
        )
        # p has shape: (n_samples, 3)

        ring_templates_list.append(p)

    # Concatenate all ring templates along the sample dimension.
    ring_templates = np.concatenate(
        ring_templates_list, axis=0
    )  # shape: (total_points, 3)

    # --- Step 2: Rotate the concatenated ring template for each center ---
    # Assume the existence of a helper function `cartesian2polar` that converts
    # Cartesian coordinates (N,3) into polar (θ, φ) with shape (N,2).
    center_polar = cartesian2polar(center_vector)  # shape: (N, 2)
    # Use -θ for the y-rotation and φ for the z-rotation.
    y_angle = -center_polar[:, 0]  # shape: (N,)
    z_angle = center_polar[:, 1]  # shape: (N,)

    # Compute the sines and cosines.
    cos_y = np.cos(y_angle)
    sin_y = np.sin(y_angle)
    cos_z = np.cos(z_angle)
    sin_z = np.sin(z_angle)

    # Construct the rotation matrices about the y-axis for each center.
    rotate_y = np.stack(
        [
            np.stack([cos_y, np.zeros_like(cos_y), sin_y], axis=-1),
            np.stack(
                [np.zeros_like(cos_y), np.ones_like(cos_y), np.zeros_like(cos_y)],
                axis=-1,
            ),
            np.stack([-sin_y, np.zeros_like(cos_y), cos_y], axis=-1),
        ],
        axis=1,
    )  # shape: (N, 3, 3)

    # Construct the rotation matrices about the z-axis for each center.
    rotate_z = np.stack(
        [
            np.stack([cos_z, -sin_z, np.zeros_like(cos_z)], axis=-1),
            np.stack([sin_z, cos_z, np.zeros_like(cos_z)], axis=-1),
            np.stack(
                [np.zeros_like(cos_z), np.zeros_like(cos_z), np.ones_like(cos_z)],
                axis=-1,
            ),
        ],
        axis=1,
    )  # shape: (N, 3, 3)

    # Total rotation for each center is given by R = rotate_z @ rotate_y.
    R = np.matmul(rotate_z, rotate_y)  # shape: (N, 3, 3)

    # Apply the rotation: for each center, we compute
    # rotated_center = ring_templates @ (R_i)^T.
    # Use broadcasting: ring_templates has shape (total_points, 3) and we expand to (1, total_points, 3).
    rotated = np.matmul(ring_templates[None, :, :], R.transpose(0, 2, 1))
    # rotated has shape: (N, total_points, 3)

    return rotated
