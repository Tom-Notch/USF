#!/usr/bin/env python3
#
# Created on Thu Mar 06 2025 06:15:33
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import matplotlib.pyplot as plt
import torch

from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.torch_numpy import to_numpy


def _gnomonic_rotate_points(
    vector: torch.Tensor,
    neg_phi: float,
    theta: float,
    neg_gamma: float,
) -> torch.Tensor:
    """
    Rotate candidate points (in Cartesian coordinates) by the inverse rotation:
    first about the z-axis by neg_phi, then about the y-axis by theta,
    and finally about the x-axis by neg_gamma.

    Note: The parameter naming is confusing; please see the convention under docs/ folder.

    Args:
        vector (torch.Tensor): Tensor of shape (N, 3) containing Cartesian points.
        neg_phi (float): Negative of the RBFoV's phi.
        theta (float): The RBFoV's theta.
        neg_gamma (float): Negative of the RBFoV's gamma.

    Returns:
        torch.Tensor: Rotated points of shape (N, 3).
    """
    _device = vector.device
    _dtype = vector.dtype

    # Ensure parameters are tensors.
    if not isinstance(neg_phi, torch.Tensor):
        neg_phi = torch.tensor(neg_phi, device=_device, dtype=_dtype)
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, device=_device, dtype=_dtype)
    if not isinstance(neg_gamma, torch.Tensor):
        neg_gamma = torch.tensor(neg_gamma, device=_device, dtype=_dtype)

    # Using torch functions to build rotation matrices without in-place operations.
    cos_z = torch.cos(neg_phi)
    sin_z = torch.sin(neg_phi)
    zero = torch.zeros_like(cos_z)
    one = torch.ones_like(cos_z)
    R_z = torch.stack(
        [
            torch.stack([cos_z, -sin_z, zero], dim=-1),
            torch.stack([sin_z, cos_z, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )  # shape (..., 3, 3)

    cos_y = torch.cos(theta)
    sin_y = torch.sin(theta)
    R_y = torch.stack(
        [
            torch.stack([cos_y, zero, sin_y], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin_y, zero, cos_y], dim=-1),
        ],
        dim=-2,
    )

    cos_x = torch.cos(neg_gamma)
    sin_x = torch.sin(neg_gamma)
    R_x = torch.stack(
        [
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, cos_x, -sin_x], dim=-1),
            torch.stack([zero, sin_x, cos_x], dim=-1),
        ],
        dim=-2,
    )

    # Combined rotation: R = R_x @ R_y @ R_z.
    R = R_x @ R_y @ R_z
    # (R @ vector.T).T
    rotated_vector = (R @ vector.mT).mT
    return rotated_vector


def extract_inside_mask(vector: torch.Tensor, rbfov: torch.Tensor) -> torch.Tensor:
    """Compute a boolean mask indicating which unit vectors fall inside a given RBFoV.

    Projects each vector via gnomonic projection centered at the RBFoV's optical axis
    and checks whether the projected point lies within the rectangular FoV boundary.

    Args:
        vector (torch.Tensor): Unit Cartesian vectors, shape (N, 3).
        rbfov (torch.Tensor): RBFoV parameters [θ, φ, α, β, γ, ...], at least 5 elements.
            θ (float): latitude of optical axis center.
            φ (float): longitude of optical axis center.
            α (float): vertical half-FoV angle (full height = 2α).
            β (float): horizontal half-FoV angle (full width = 2β).
            γ (float): roll angle of the camera.

    Returns:
        torch.Tensor: Boolean mask of shape (N,), True where the vector is inside the RBFoV.
    """
    theta, phi, alpha, beta, gamma = rbfov[..., :5]

    # gnomonic rotate to polar = (0, 0) or vector = (1, 0, 0), including gamma
    gnomonic_rotated_vecs = _gnomonic_rotate_points(
        vector=vector,
        neg_phi=-phi,
        theta=theta,
        neg_gamma=-gamma,
    )

    # gnomonic projection
    gnomonic_y = gnomonic_rotated_vecs[..., 1] / gnomonic_rotated_vecs[..., 0]  # y/x
    gnomonic_z = gnomonic_rotated_vecs[..., 2] / gnomonic_rotated_vecs[..., 0]  # z/x

    # define boundary
    half_width, half_height = torch.tan(beta / 2), torch.tan(alpha / 2)

    # x must be positive to have a valid gnomonic projection
    inside_mask = (
        (gnomonic_y.abs() < half_width)
        & (gnomonic_z.abs() < half_height)
        & (gnomonic_rotated_vecs[..., 0] > 1e-6)
    )

    return inside_mask


def pairwiseIoU(
    rbfovs1: torch.Tensor,
    rbfovs2: torch.Tensor,
    vector_average_area: float = 2.727076956241019e-05,  # from input spherical image
    location_sampler: str = "icosahedron",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute approximate pairwise intersection over union between each pair of (rbfov1, rbfov2), outputs a confusion matrix with element the IoU value.
    This is done by counting the number of vector inside an rbfov/intersection/union using boolean mask operations.
    We will first use location sampler to sample an approximately uniformly distributed unit vectors on the sphere, then determine masks for rbfovs1 and rbfovs2 in a vectorized way.
    Next, we will construct 2 integer matrices: intersection count and union count.
    Finally, we can get the pairwise IoU matrix by dividing the intersection count matrix by union count matrix.
    Each RBFoV's format: [θ, φ, α, β, γ, category(, confidence)]

    Args:
        rbfovs1 (torch.Tensor): First set of RBFoVs, shape (N, 5+).
        rbfovs2 (torch.Tensor): Second set of RBFoVs, shape (M, 5+).
        vector_average_area (float, optional): The average vector area on the sphere for the vector to sample from location_sampler
        location_sampler (str, optional): type of location sampler to sample "uniform" vector from
        eps (float, optional): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: a confusion matrix if both rbfovs are non-empty, shape (rbfovs1.shape[0], rbfovs2.shape[0])
    """
    assert (
        0.0 < vector_average_area < 4 * torch.pi
    ), f"Expect vector_average_area to be in (0.0, 4 x pi), got {vector_average_area}"

    # avoid accidental modification to input
    _rbfovs1, _rbfovs2 = rbfovs1.clone(), rbfovs2.clone()
    _device, _dtype = _rbfovs1.device, _rbfovs1.dtype

    # sample "uniform vector"
    vector = LocationSampler({"location_sampler": location_sampler})(
        vector_average_area
    ).to(device=_device, dtype=_dtype)

    # stack rbfovs when creating masks
    _rbfovs = torch.vstack((_rbfovs1, _rbfovs2))

    # get inside masks
    inside_masks = torch.vmap(extract_inside_mask)(
        vector.unsqueeze(0).expand(_rbfovs.shape[0], vector.shape[-2], 3), _rbfovs
    )

    N = _rbfovs1.shape[0]
    mask1 = inside_masks[:N, :]  # shape: (N, K)
    mask2 = inside_masks[N:, :]  # shape: (M, K)

    # Compute the area (number of candidate points inside each RBFoV).
    area1 = mask1.float().sum(dim=1)  # shape: (N,)
    area2 = mask2.float().sum(dim=1)  # shape: (M,)

    # Compute the intersection counts (via matrix multiplication on the binary masks).
    intersection = torch.matmul(mask1.float(), mask2.float().T)  # shape: (N, M)

    # Compute union: union = area1 + area2 - intersection (broadcasting correctly).
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection  # shape: (N, M)

    iou_matrix = intersection / (union + eps)

    return iou_matrix


def visualize_iou_confusion(
    iou_matrix: torch.Tensor,
    title: str = "IoU Confusion Matrix",
) -> None:
    """Display a heatmap of a pairwise IoU confusion matrix using matplotlib.

    Args:
        iou_matrix (torch.Tensor): IoU confusion matrix of shape (N, M).
        title (str, optional): Title of the plot. Defaults to "IoU Confusion Matrix".
    """
    iou_np = to_numpy(iou_matrix)

    plt.figure(figsize=(8, 6))
    plt.imshow(iou_np, cmap="viridis", interpolation="none")
    plt.title(title)
    plt.xlabel("GT RBFoVs")
    plt.ylabel("Predicted RBFoVs")
    plt.colorbar(label="IoU")
    plt.show()
