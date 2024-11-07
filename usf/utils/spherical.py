#!/usr/bin/env python3
#
# Created on Fri Nov 22 2024 11:59:00
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from multimethod import multimethod
from scipy.spatial import SphericalVoronoi

from usf.utils.nearest_neighbor import nearest_point
from usf.utils.torch_numpy import copy_or_clone, to_numpy, to_torch


def unique_rows_preserve_order(x: torch.Tensor, decimals: int = 8) -> torch.Tensor:
    """Remove duplicate rows while preserving the order of first occurrence.

    Rows are considered duplicates when they agree to ``decimals`` decimal
    places after rounding.

    Args:
        x (torch.Tensor): Input tensor of shape ``(N, d)``.
        decimals (int, optional): Number of decimal places used for rounding
            before comparison. Defaults to 8.

    Returns:
        torch.Tensor: Deduplicated tensor whose rows appear in the same order
            as their first occurrence in ``x``.
    """
    x = x.round(decimals=decimals)
    # Get unique rows (arbitrary order) + mapping from each row → unique id
    values, inverse = torch.unique(x, dim=0, return_inverse=True, sorted=False)

    # For each unique id, find the first index where it appeared
    N = x.shape[0]
    first_idx = torch.full((values.shape[0],), N, device=x.device, dtype=torch.long)
    first_idx.scatter_reduce_(
        0,
        inverse,
        torch.arange(N, device=x.device),
        reduce="amin",
    )

    # Reorder uniques by their first appearance in the input
    order = torch.argsort(first_idx)
    return values[order]


@multimethod
def compute_inside_fov_mask(  # type: ignore
    input_locations: np.ndarray,
    target_locations: np.ndarray,
    distance_threshold_scaling: float = 2.0,
    close_count_threshold: int = 3,
) -> np.ndarray:
    """Compute a boolean mask indicating which input locations lie inside the FOV (numpy).

    A point is considered inside if it is close to at least ``close_count_threshold``
    target directions, within a distance threshold defined by
    ``distance_threshold_scaling * max_nearest_neighbor_distance``.

    Args:
        input_locations (np.ndarray): Unit-length Cartesian vectors to test,
            of shape ``(N, 3)``.
        target_locations (np.ndarray): Reference unit-length Cartesian vectors
            defining the field of view (e.g. equirectangular image pixels),
            of shape ``(M, 3)``.
        distance_threshold_scaling (float, optional): Multiplier applied to the
            maximum nearest-neighbor distance among target locations to derive the
            angular threshold. Defaults to 2.0.
        close_count_threshold (int, optional): Minimum number of target neighbors
            within the threshold required to consider a point "inside". Defaults to 3.

    Returns:
        np.ndarray: Boolean mask of shape ``(N,)`` where ``True`` means the
            corresponding input location is inside the FOV.
    """
    # Distance threshold to determine "inside" the spherical image
    angular_distance_threshold = (
        nearest_neighbor_distance(target_locations).max() * distance_threshold_scaling
    )

    # A valid sample location needs to be close to at least close_count_threshold image pixel location(s)
    _, angles = nearest_point(input_locations, target_locations, close_count_threshold)
    close_counts = np.sum(angles <= angular_distance_threshold, axis=-1)
    return close_counts >= close_count_threshold


@multimethod
def compute_inside_fov_mask(
    input_locations: torch.Tensor,
    target_locations: torch.Tensor,
    distance_threshold_scaling: float = 2.0,
    close_count_threshold: int = 3,
) -> torch.Tensor:
    """Compute a boolean mask indicating which input locations lie inside the FOV (torch).

    A point is considered inside if it is close to at least ``close_count_threshold``
    target directions, within a distance threshold defined by
    ``distance_threshold_scaling * max_nearest_neighbor_distance``.

    Args:
        input_locations (torch.Tensor): Unit-length Cartesian vectors to test,
            of shape ``(N, 3)``.
        target_locations (torch.Tensor): Reference unit-length Cartesian vectors
            defining the field of view (e.g. equirectangular image pixels),
            of shape ``(M, 3)``.
        distance_threshold_scaling (float, optional): Multiplier applied to the
            maximum nearest-neighbor distance among target locations to derive the
            angular threshold. Defaults to 2.0.
        close_count_threshold (int, optional): Minimum number of target neighbors
            within the threshold required to consider a point "inside". Defaults to 3.

    Returns:
        torch.Tensor: Boolean mask of shape ``(N,)`` where ``True`` means the
            corresponding input location is inside the FOV.
    """
    # Distance threshold to determine "inside" the spherical image
    angular_distance_threshold = (
        nearest_neighbor_distance(target_locations).max() * distance_threshold_scaling
    )

    # A valid sample location needs to be close to at least close_count_threshold image pixel location(s)
    _, angles = nearest_point(input_locations, target_locations, close_count_threshold)
    close_counts = torch.sum(angles <= angular_distance_threshold, dim=-1)
    return close_counts >= close_count_threshold


@multimethod
def relative_direction(  # type: ignore
    reference_vector: torch.Tensor, input_vector: torch.Tensor
) -> torch.Tensor:
    """Compute the signed angular direction of ``input_vector`` relative to ``reference_vector`` (torch).

    The angle is measured in the tangent plane at ``reference_vector``, with
    "north" defined by the global +z axis projected onto that plane. Sign follows
    the right-hand rule: positive when ``input_vector`` lies to the left of
    ``reference_vector`` when looking toward the origin.

    Args:
        reference_vector (torch.Tensor): Reference unit vectors of shape ``(..., 3)``.
        input_vector (torch.Tensor): Input unit vectors of shape ``(..., 3)``;
            must share the same shape, device, and dtype as ``reference_vector``.

    Returns:
        torch.Tensor: Signed relative direction in radians, shape ``(...,)``,
            in the range ``[-pi, pi]``.
    """
    assert (
        reference_vector.shape == input_vector.shape
        and reference_vector.device == input_vector.device
        and reference_vector.dtype == input_vector.dtype
    ), "reference_vector and input_vector must have the same shape, device, and dtype."

    _device = reference_vector.device
    _dtype = reference_vector.dtype

    # Project input_vector onto tangent plane of reference_vector.
    dot = (reference_vector * input_vector).sum(dim=-1, keepdim=True)
    input_tangent_vector = input_vector - dot * reference_vector  # tangent component
    input_tangent_norm_vector = F.normalize(input_tangent_vector, p=2, dim=-1)

    # Define global north.
    global_north = torch.zeros_like(reference_vector, device=_device, dtype=_dtype)
    global_north[..., -1] = 1  # set z to 1 for north

    # For each candidate, project global_north onto tangent plane at reference_vector:
    dot_north = (reference_vector * global_north).sum(dim=-1, keepdim=True)
    north_tangent_vector = global_north - dot_north * reference_vector
    north_tangent_norm_vector = F.normalize(north_tangent_vector, p=2, dim=-1)

    # Compute unsigned angle between input_tangent_norm_vector and north_tangent_norm_vector.
    dot_nv = (input_tangent_norm_vector * north_tangent_norm_vector).sum(dim=-1)
    unsigned_angle = torch.acos(torch.clamp(dot_nv, -1.0, 1.0))  # in [0, pi]

    # Determine sign: compute cross(north_tangent_norm_vector, input_tangent_norm_vector) and dot with reference_vector.
    cross_nv = torch.cross(north_tangent_norm_vector, input_tangent_norm_vector, dim=-1)
    sign = torch.sign((reference_vector * cross_nv).sum(dim=-1))
    direction = unsigned_angle * sign  # in [-pi, pi]

    return direction


@multimethod
def relative_direction(
    reference_vector: np.ndarray, input_vector: np.ndarray
) -> np.ndarray:
    """Compute the signed angular direction of ``input_vector`` relative to ``reference_vector`` (numpy).

    The angle is measured in the tangent plane at ``reference_vector``, with
    "north" defined by the global +z axis projected onto that plane. Sign follows
    the right-hand rule: positive when ``input_vector`` lies to the left of
    ``reference_vector`` when looking toward the origin.

    Args:
        reference_vector (np.ndarray): Reference unit vectors of shape ``(..., 3)``.
        input_vector (np.ndarray): Input unit vectors of shape ``(..., 3)``;
            must share the same shape as ``reference_vector``.

    Returns:
        np.ndarray: Signed relative direction in radians, shape ``(...,)``,
            in the range ``[-pi, pi]``.
    """
    # Ensure shapes match.
    assert (
        reference_vector.shape == input_vector.shape
    ), "Shapes of reference_vector and input_vector must match."

    # Project input_vector onto the tangent plane of reference_vector.
    dot = np.sum(reference_vector * input_vector, axis=-1, keepdims=True)  # (..., 1)
    input_tangent_vector = input_vector - dot * reference_vector  # tangent component
    norm = np.linalg.norm(input_tangent_vector, axis=-1, keepdims=True) + 1e-6
    input_tangent_norm_vector = input_tangent_vector / norm  # normalized tangent

    # Define global north as [0,0,1] with the same shape as reference_vector.
    global_north = np.zeros_like(reference_vector)
    global_north[..., -1] = 1.0

    # Project global_north onto the tangent plane at reference_vector.
    dot_north = np.sum(reference_vector * global_north, axis=-1, keepdims=True)
    north_tangent_vector = global_north - dot_north * reference_vector
    norm_north = np.linalg.norm(north_tangent_vector, axis=-1, keepdims=True) + 1e-6
    north_tangent_norm_vector = north_tangent_vector / norm_north

    # Compute the unsigned angle between input_tangent_norm_vector and north_tangent_norm_vector.
    dot_nv = np.sum(input_tangent_norm_vector * north_tangent_norm_vector, axis=-1)
    dot_nv = np.clip(dot_nv, -1.0, 1.0)
    unsigned_angle = np.arccos(dot_nv)  # in [0, pi]

    # Compute the cross product and determine the sign.
    cross_nv = np.cross(north_tangent_norm_vector, input_tangent_norm_vector)
    sign = np.sign(np.sum(reference_vector * cross_nv, axis=-1))
    direction = unsigned_angle * sign  # in [-pi, pi]

    return direction


@multimethod
def azim_elev_to_rotation_matrix(azimuth_elevation: torch.Tensor) -> torch.Tensor:  # type: ignore
    """Build a batch of SO(3) rotation matrices from azimuth and elevation angles (torch).

    The rotation is composed as ``R_z(azimuth) @ R_y(-elevation)``, i.e. yaw
    followed by pitch.

    Args:
        azimuth_elevation (torch.Tensor): Azimuth and elevation angles in radians,
            of shape ``(..., 2)``. The last dimension encodes
            ``[azimuth (yaw), elevation (pitch)]``.

    Returns:
        torch.Tensor: Batch of 3x3 rotation matrices of shape ``(..., 3, 3)``.
    """
    assert (
        azimuth_elevation.shape[-1] == 2
    ), "Each azimuth_elevation should have 2 values (azimuth, elevation)"

    _azimuth_elevation = copy_or_clone(azimuth_elevation)
    device, dtype = _azimuth_elevation.device, _azimuth_elevation.dtype

    azimuth, elevation = _azimuth_elevation[..., 0], _azimuth_elevation[..., 1]
    sin_z, cos_z = torch.sin(azimuth), torch.cos(azimuth)
    sin_y, cos_y = torch.sin(-elevation), torch.cos(-elevation)
    zero = torch.zeros_like(azimuth, dtype=dtype, device=device)
    one = torch.ones_like(azimuth, dtype=dtype, device=device)

    R_z = torch.stack(
        [
            torch.stack([cos_z, -sin_z, zero], dim=-1),
            torch.stack([sin_z, cos_z, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ],
        dim=-2,
    )  # shape (..., 3, 3)

    R_y = torch.stack(
        [
            torch.stack([cos_y, zero, sin_y], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin_y, zero, cos_y], dim=-1),
        ],
        dim=-2,
    )

    return R_z @ R_y


@multimethod
def azim_elev_to_rotation_matrix(azimuth_elevation: np.ndarray) -> np.ndarray:
    """Build a batch of SO(3) rotation matrices from azimuth and elevation angles (numpy).

    The rotation is composed as ``R_z(azimuth) @ R_y(-elevation)``, i.e. yaw
    followed by pitch.

    Args:
        azimuth_elevation (np.ndarray): Azimuth and elevation angles in radians,
            of shape ``(..., 2)``. The last dimension encodes
            ``[azimuth (yaw), elevation (pitch)]``.

    Returns:
        np.ndarray: Batch of 3x3 rotation matrices of shape ``(..., 3, 3)``.
    """
    assert (
        azimuth_elevation.shape[-1] == 2
    ), "Each azimuth_elevation should have 2 values (azimuth, elevation)"

    _azimuth_elevation = copy_or_clone(azimuth_elevation)

    azimuth, elevation = _azimuth_elevation[..., 0], _azimuth_elevation[..., 1]
    sin_z, cos_z = np.sin(azimuth), np.cos(azimuth)
    sin_y, cos_y = np.sin(-elevation), np.cos(-elevation)
    zero = np.zeros_like(azimuth)
    one = np.ones_like(azimuth)

    R_z = np.stack(
        [
            np.stack([cos_z, -sin_z, zero], axis=-1),
            np.stack([sin_z, cos_z, zero], axis=-1),
            np.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )  # shape (..., 3, 3)

    R_y = np.stack(
        [
            np.stack([cos_y, zero, sin_y], axis=-1),
            np.stack([zero, one, zero], axis=-1),
            np.stack([-sin_y, zero, cos_y], axis=-1),
        ],
        axis=-2,
    )

    return R_z @ R_y


@multimethod
def normalize_cartesian(vector: np.ndarray) -> np.ndarray:  # type: ignore
    """Normalize Cartesian vectors to unit length (numpy).

    Args:
        vector (np.ndarray): Array of shape ``(..., 3)`` containing Cartesian
            coordinates ``(x, y, z)``. Zero vectors are left unchanged (norm is
            treated as 1 to avoid division by zero).

    Returns:
        np.ndarray: Array of shape ``(..., 3)`` containing L2-normalized
            Cartesian coordinates.
    """
    assert (
        vector.shape[-1] == 3
    ), "Each cartesian coordinate should have 3 values (x, y, z)"

    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)  # avoid division by 0

    return vector / norm


@multimethod
def normalize_cartesian(vector: torch.Tensor) -> torch.Tensor:
    """Normalize Cartesian vectors to unit length (torch).

    Args:
        vector (torch.Tensor): Tensor of shape ``(..., 3)`` containing Cartesian
            coordinates ``(x, y, z)``.

    Returns:
        torch.Tensor: Tensor of shape ``(..., 3)`` containing L2-normalized
            Cartesian coordinates.
    """
    assert (
        vector.shape[-1] == 3
    ), "Each cartesian coordinate should have 3 values (x, y, z)"

    return F.normalize(vector, p=2, dim=-1)


@multimethod
def normalize_polar(polar: np.ndarray) -> np.ndarray:  # type: ignore
    """Wrap polar coordinates into canonical ranges (numpy).

    Casts theta (latitude) into ``[-pi/2, pi/2]`` and phi (longitude) into
    ``[-pi, pi]``. When theta falls in the "over-rotated" band ``(pi/2, pi]``,
    it is reflected and phi is shifted by ``pi`` to represent the same direction.

    Args:
        polar (np.ndarray): Array of shape ``(..., 2)`` containing polar
            coordinates ``(theta, phi)`` in radians, where theta is latitude
            and phi is longitude.

    Returns:
        np.ndarray: Array of shape ``(..., 2)`` containing normalized polar
            coordinates with theta in ``[-pi/2, pi/2]`` and phi in ``[-pi, pi]``.
    """

    # input validation
    assert (
        polar.shape[-1] == 2
    ), "Each polar coordinate should have 2 values theta and phi"

    # pre-processing
    polar_flat = polar.reshape(-1, 2).copy()

    theta = polar_flat[:, 0]
    phi = polar_flat[:, 1]

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    theta = theta % (2 * np.pi)
    theta[theta >= 3 * np.pi / 2] -= 2 * np.pi

    # if the original theta resides in [pi/2, pi] or [-pi, -pi/2], perform proper rotation on both theta and phi
    over_rotated_idx = np.argwhere(theta > np.pi / 2)
    theta[over_rotated_idx] = np.pi - theta[over_rotated_idx]
    phi[over_rotated_idx] += np.pi

    phi = phi % (2 * np.pi)
    phi[phi > np.pi] -= 2 * np.pi

    # inplace update
    polar = np.hstack((theta[:, None], phi[:, None])).reshape(*polar.shape)

    return polar


@multimethod
def normalize_polar(polar: torch.Tensor) -> torch.Tensor:
    """Wrap polar coordinates into canonical ranges (torch).

    Casts theta (latitude) into ``[-pi/2, pi/2]`` and phi (longitude) into
    ``[-pi, pi]``. When theta falls in the "over-rotated" band ``(pi/2, pi]``,
    it is reflected and phi is shifted by ``pi`` to represent the same direction.

    Args:
        polar (torch.Tensor): Tensor of shape ``(..., 2)`` containing polar
            coordinates ``(theta, phi)`` in radians, where theta is latitude
            and phi is longitude.

    Returns:
        torch.Tensor: Tensor of shape ``(..., 2)`` containing normalized polar
            coordinates with theta in ``[-pi/2, pi/2]`` and phi in ``[-pi, pi]``.
    """

    # input validation
    assert (
        polar.shape[-1] == 2
    ), "Each polar coordinate should have 2 values theta and phi"

    # pre-processing
    polar_flat = polar.clone().view(-1, 2)

    theta = polar_flat[:, 0]
    phi = polar_flat[:, 1]

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    theta = theta % (2 * torch.pi)
    theta[theta >= 3 * torch.pi / 2] -= 2 * torch.pi

    # if the original theta resides in [pi/2, pi] or [-pi, -pi/2], perform proper rotation on both theta and phi
    over_rotated_idx = torch.argwhere(theta > torch.pi / 2)
    theta[over_rotated_idx] = torch.pi - theta[over_rotated_idx]
    phi[over_rotated_idx] += torch.pi

    phi = phi % (2 * torch.pi)
    phi[phi > torch.pi] -= 2 * torch.pi

    # inplace update
    polar = torch.hstack((theta[:, None], phi[:, None])).view(*polar.shape)

    return polar


@multimethod
def polar2cartesian(polar: np.ndarray) -> np.ndarray:  # type: ignore
    """Convert polar coordinates to unit Cartesian vectors (numpy).

    Polar coordinates are first normalized to canonical ranges via
    :func:`normalize_polar` before conversion.

    Args:
        polar (np.ndarray): Array of shape ``(..., 2)`` containing polar
            coordinates ``(theta, phi)`` in radians, where theta is latitude
            in ``[-pi/2, pi/2]`` and phi is longitude in ``[-pi, pi]``.

    Returns:
        np.ndarray: Array of shape ``(..., 3)`` containing unit Cartesian
            vectors ``(x, y, z)`` where x points forward, y points left,
            and z points up.
    """

    # input validation
    assert (
        polar.shape[-1] == 2
    ), "Each polar coordinate should have 2 values theta and phi"

    # pre-processing
    polar_flat = polar.reshape(-1, 2).copy()

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    polar_flat = normalize_polar(polar_flat)

    theta = polar_flat[:, 0]
    phi = polar_flat[:, 1]

    # compute the co-centric unit 3D vector
    x = np.cos(theta) * np.cos(phi)  # x = cos(theta) * cos(phi)
    y = np.cos(theta) * np.sin(phi)  # y = cos(theta) * sin(phi)
    z = np.sin(theta)  # z = sin(theta)

    vector = np.hstack((x[:, None], y[:, None], z[:, None]))

    # normalize the vector
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)

    return vector.reshape((*polar.shape[:-1], 3))


@multimethod
def polar2cartesian(polar: torch.Tensor) -> torch.Tensor:
    """Convert polar coordinates to unit Cartesian vectors (torch).

    Polar coordinates are first normalized to canonical ranges via
    :func:`normalize_polar` before conversion.

    Args:
        polar (torch.Tensor): Tensor of shape ``(..., 2)`` containing polar
            coordinates ``(theta, phi)`` in radians, where theta is latitude
            in ``[-pi/2, pi/2]`` and phi is longitude in ``[-pi, pi]``.

    Returns:
        torch.Tensor: Tensor of shape ``(..., 3)`` containing unit Cartesian
            vectors ``(x, y, z)`` where x points forward, y points left,
            and z points up.
    """

    # input validation
    assert (
        polar.shape[-1] == 2
    ), "Each polar coordinate should have 2 values theta and phi"

    # pre-processing
    polar_flat = polar.reshape(-1, 2).clone()

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    polar_flat = normalize_polar(polar_flat)

    theta = polar_flat[:, 0]
    phi = polar_flat[:, 1]

    # compute the co-centric unit 3D vector
    x = torch.cos(theta) * torch.cos(phi)  # x = cos(theta) * cos(phi)
    y = torch.cos(theta) * torch.sin(phi)  # y = cos(theta) * sin(phi)
    z = torch.sin(theta)  # z = sin(theta)

    vector = torch.hstack((x[:, None], y[:, None], z[:, None]))

    # normalize the vector
    vector /= torch.linalg.norm(vector, dim=1, keepdims=True)

    return vector.view((*polar.shape[:-1], 3))


@multimethod
def cartesian2polar(v: np.ndarray) -> np.ndarray:  # type: ignore
    """Convert co-centric 3D vectors to polar coordinates (numpy).

    The input does not need to be unit-length; vectors are normalized before
    conversion. The resulting polar coordinates are in canonical form via
    :func:`normalize_polar`.

    Args:
        v (np.ndarray): Array of shape ``(..., 3)`` containing co-centric 3D
            vectors ``(x, y, z)`` where x points forward, y points left, and z
            points up.

    Returns:
        np.ndarray: Array of shape ``(..., 2)`` containing polar coordinates
            ``(theta, phi)`` in radians, where theta is latitude in
            ``[-pi/2, pi/2]`` and phi is longitude in ``[-pi, pi]``.
    """

    # input validation
    assert v.shape[-1] == 3, "Each vector should have 3 values x, y, and z"

    # pre-processing
    v_flat = v.reshape(-1, 3).astype(v.dtype, copy=True)
    v_flat /= np.linalg.norm(v_flat, axis=1, keepdims=True)

    x = v_flat[:, 0]
    y = v_flat[:, 1]
    z = v_flat[:, 2]

    # compute the polar coordinates
    theta = np.arctan2(z, np.sqrt(x**2 + y**2))  # theta = arctan(z/sqrt(x^2 + y^2))
    phi = np.arctan2(y, x)  # phi = arctan(y/x)

    polar = np.hstack((theta[:, None], phi[:, None]))

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    polar = normalize_polar(polar)

    return polar.reshape((*v.shape[:-1], 2))


@multimethod
def cartesian2polar(v: torch.Tensor) -> torch.Tensor:
    """Convert co-centric 3D vectors to polar coordinates (torch).

    The input does not need to be unit-length; vectors are normalized before
    conversion. The resulting polar coordinates are in canonical form via
    :func:`normalize_polar`.

    Args:
        v (torch.Tensor): Tensor of shape ``(..., 3)`` containing co-centric 3D
            vectors ``(x, y, z)`` where x points forward, y points left, and z
            points up.

    Returns:
        torch.Tensor: Tensor of shape ``(..., 2)`` containing polar coordinates
            ``(theta, phi)`` in radians, where theta is latitude in
            ``[-pi/2, pi/2]`` and phi is longitude in ``[-pi, pi]``.
    """

    # input validation
    assert v.shape[-1] == 3, "Each vector should have 3 values x, y, and z"

    # pre-processing
    v_flat = v.reshape(-1, 3).clone()
    v_flat /= torch.linalg.norm(v_flat, dim=1, keepdims=True)

    x = v_flat[:, 0]
    y = v_flat[:, 1]
    z = v_flat[:, 2]

    # compute the polar coordinates
    theta = torch.arctan2(
        z, torch.sqrt(x**2 + y**2)
    )  # theta = arctan(z/sqrt(x^2 + y^2))
    phi = torch.arctan2(y, x)  # phi = arctan(y/x)

    polar = torch.hstack((theta[:, None], phi[:, None]))

    # cast theta to [-pi/2, pi/2] and phi to [-pi, pi]
    polar = normalize_polar(polar)

    return polar.view((*v.shape[:-1], 2))


def sample_random_rotation_matrices(batch_size: int = 1) -> torch.Tensor:
    """Generate uniformly random rotation matrices in SO(3) using unit quaternions.

    Quaternions are sampled from a 4D standard normal distribution and normalized
    to the unit sphere, which gives a uniform distribution over SO(3).

    Args:
        batch_size (int, optional): Number of random rotation matrices to generate.
            Defaults to 1.

    Returns:
        torch.Tensor: Tensor of shape ``(batch_size, 3, 3)`` where each
            sub-matrix is an orthonormal rotation matrix.
    """
    # Step 1: sample (batch_size x 4) standard normal random values
    q = torch.randn(batch_size, 4)

    # Step 2: normalize to get unit quaternions
    q = q / q.norm(dim=1, keepdim=True)

    # Separate out components for readability
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    # Step 3: Convert each quaternion to a 3x3 rotation matrix
    # Shape will be (batch_size, 3, 3)
    R = torch.empty((batch_size, 3, 3))

    R[:, 0, 0] = 1 - 2 * (q2**2 + q3**2)
    R[:, 0, 1] = 2 * (q1 * q2 - q0 * q3)
    R[:, 0, 2] = 2 * (q1 * q3 + q0 * q2)

    R[:, 1, 0] = 2 * (q1 * q2 + q0 * q3)
    R[:, 1, 1] = 1 - 2 * (q1**2 + q3**2)
    R[:, 1, 2] = 2 * (q2 * q3 - q0 * q1)

    R[:, 2, 0] = 2 * (q1 * q3 - q0 * q2)
    R[:, 2, 1] = 2 * (q2 * q3 + q0 * q1)
    R[:, 2, 2] = 1 - 2 * (q1**2 + q2**2)

    return R


@multimethod
def spherical_distance(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:  # type: ignore
    """Compute the great-circle (spherical) distance between pairs of points (numpy).

    Each point may be given as polar coordinates ``(theta, phi)`` or as a
    co-centric 3D Cartesian vector; both forms are accepted and mixed within
    a single call.

    Args:
        p1 (np.ndarray): Array of shape ``(..., 2)`` for polar coordinates or
            ``(..., 3)`` for Cartesian vectors. Both arrays must contain the
            same number of points.
        p2 (np.ndarray): Array of shape ``(..., 2)`` for polar coordinates or
            ``(..., 3)`` for Cartesian vectors. Must match ``p1`` in number of
            points.

    Returns:
        np.ndarray: Great-circle distance in radians, shape matching
            ``p1.shape[:-1]``.
    """

    # input validation
    assert (p1.shape[-1] == 2 or p1.shape[-1] == 3) and (
        p2.shape[-1] == 2 or p2.shape[-1] == 3
    ), "Each point should have 2 values theta and phi, or 3 values x, y, and z"

    # pre-processing without touching the original data, in the spirit of pure functional programming
    s1_flat = (
        normalize_polar(p1.copy()) if p1.shape[-1] == 2 else cartesian2polar(p1.copy())
    ).reshape(-1, 2)
    polar_flat = (
        normalize_polar(p2.copy()) if p2.shape[-1] == 2 else cartesian2polar(p2.copy())
    ).reshape(-1, 2)

    assert (
        s1_flat.shape == polar_flat.shape
    ), "The two input should have same number of points"

    theta1 = s1_flat[:, 0]
    phi1 = s1_flat[:, 1]
    theta2 = polar_flat[:, 0]
    phi2 = polar_flat[:, 1]

    # compute the spherical distance
    d = np.arccos(
        np.clip(
            np.sin(theta1) * np.sin(theta2)
            + np.cos(theta1) * np.cos(theta2) * np.cos(phi1 - phi2),
            -1.0,
            1.0,
        )
    )

    return d.reshape(p1.shape[:-1])


@multimethod
def spherical_distance(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Compute the great-circle (spherical) distance between pairs of points (torch).

    Each point may be given as polar coordinates ``(theta, phi)`` or as a
    co-centric 3D Cartesian vector; both forms are accepted and mixed within
    a single call.

    Args:
        p1 (torch.Tensor): Tensor of shape ``(..., 2)`` for polar coordinates or
            ``(..., 3)`` for Cartesian vectors. Both tensors must contain the
            same number of points.
        p2 (torch.Tensor): Tensor of shape ``(..., 2)`` for polar coordinates or
            ``(..., 3)`` for Cartesian vectors. Must match ``p1`` in number of
            points.

    Returns:
        torch.Tensor: Great-circle distance in radians, shape matching
            ``p1.shape[:-1]``.
    """

    # input validation
    assert (p1.shape[-1] == 2 or p1.shape[-1] == 3) and (
        p2.shape[-1] == 2 or p2.shape[-1] == 3
    ), "Each point should have 2 values theta and phi, or 3 values x, y, and z"

    # pre-processing without touching the original data, in the spirit of pure functional programming
    s1_flat = (
        normalize_polar(p1.clone())
        if p1.shape[-1] == 2
        else cartesian2polar(p1.clone())
    ).reshape(-1, 2)
    polar_flat = (
        normalize_polar(p2.clone())
        if p2.shape[-1] == 2
        else cartesian2polar(p2.clone())
    ).reshape(-1, 2)

    assert (
        s1_flat.shape == polar_flat.shape
    ), "The two input should have same number of points"

    theta1 = s1_flat[:, 0]
    phi1 = s1_flat[:, 1]
    theta2 = polar_flat[:, 0]
    phi2 = polar_flat[:, 1]

    # compute the spherical distance
    d = torch.arccos(
        torch.clamp(
            torch.sin(theta1) * torch.sin(theta2)
            + torch.cos(theta1) * torch.cos(theta2) * torch.cos(phi1 - phi2),
            -1.0,
            1.0,
        )
    )

    return d.view(p1.shape[:-1])


@multimethod
def ripple_sort(  # type: ignore
    source: np.ndarray, target: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
) -> np.ndarray:
    """Sort coordinates by ascending spherical distance from a target point (numpy).

    The coordinate with the smallest angular distance to ``target`` appears first
    (index 0), so that index grows as points "ripple away" from the target.

    Args:
        source (np.ndarray): Coordinates to sort. Either co-centric 3D Cartesian
            vectors of shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.
        target (np.ndarray, optional): Reference point. Either a Cartesian vector
            of shape ``(3,)`` or polar coordinates of shape ``(2,)``. Defaults to
            the origin of the polar coordinate system ``[0.0, 0.0]``.

    Returns:
        np.ndarray: Sorted coordinates with the same shape as ``source``.
    """

    target = np.array(target)

    # input validation
    assert (
        source.shape[-1] == 3 or source.shape[-1] == 2
    ), "Each source vector should have 3 values x, y, and z, or 2 values theta and phi"

    assert (
        target.shape[-1] == 3 or target.shape[-1] == 2
    ), "Target vector should have 3 values x, y, and z, or 2 values theta and phi"

    # pre-processing
    source_flat = (
        source.reshape(-1, 3).copy()
        if source.shape[-1] == 3
        else source.reshape(-1, 2).copy()
    )
    target = (
        target.reshape(3).copy() if target.shape[-1] == 3 else target.reshape(2).copy()
    )

    # compute the polar coordinates
    source_spherical = (
        cartesian2polar(source_flat) if source.shape[-1] == 3 else source_flat
    )
    target_spherical = cartesian2polar(target) if target.shape[-1] == 3 else target

    # compute the spherical distance
    d = spherical_distance(
        source_spherical, np.full_like(source_spherical, target_spherical)
    )

    # sort the polar coordinates based on distance to from_vector in ascending order
    # so that vectors get away from the target when index increases
    idx = np.argsort(d)

    return source_flat[idx].reshape(source.shape)


@multimethod
def ripple_sort(
    source: torch.Tensor, target: torch.Tensor = torch.Tensor([0.0, 0.0])
) -> torch.Tensor:
    """Sort coordinates by ascending spherical distance from a target point (torch).

    The coordinate with the smallest angular distance to ``target`` appears first
    (index 0), so that index grows as points "ripple away" from the target.

    Args:
        source (torch.Tensor): Coordinates to sort. Either co-centric 3D Cartesian
            vectors of shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.
        target (torch.Tensor, optional): Reference point. Either a Cartesian vector
            of shape ``(3,)`` or polar coordinates of shape ``(2,)``. Defaults to
            the origin of the polar coordinate system ``[0.0, 0.0]``.

    Returns:
        torch.Tensor: Sorted coordinates with the same shape as ``source``.
    """

    target = target.to(device=source.device, dtype=source.dtype)

    # input validation
    assert (
        source.shape[-1] == 3 or source.shape[-1] == 2
    ), "Each source vector should have 3 values x, y, and z, or 2 values theta and phi"

    assert (
        target.shape[-1] == 3 or target.shape[-1] == 2
    ), "Target vector should have 3 values x, y, and z, or 2 values theta and phi"

    # pre-processing
    source_flat = (
        source.reshape(-1, 3).clone()
        if source.shape[-1] == 3
        else source.reshape(-1, 2).clone()
    )
    target = (
        target.reshape(3).clone()
        if target.shape[-1] == 3
        else target.reshape(2).clone()
    )

    # compute the polar coordinates
    source_spherical = (
        cartesian2polar(source_flat) if source.shape[-1] == 3 else source_flat
    )
    target_spherical = cartesian2polar(target) if target.shape[-1] == 3 else target

    # compute the spherical distance
    d = spherical_distance(
        source_spherical, target_spherical.expand_as(source_spherical)
    )

    # sort the polar coordinates based on distance to from_vector in ascending order
    # so that vectors get away from the target when index increases
    idx = torch.argsort(d, dim=0, descending=False)

    return source_flat[idx].view(source.shape)


@multimethod
def nearest_neighbor_distance(coordinate: np.ndarray) -> np.ndarray:  # type: ignore
    """Return the angular distance to the nearest distinct neighbor for each point (numpy).

    Distances are returned in radians and computed in Cartesian space via FAISS.

    Args:
        coordinate (np.ndarray): Point cloud given as unit Cartesian vectors of
            shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.

    Returns:
        np.ndarray: 1-D array of shape ``(N,)`` containing the angular distance
            in radians from each point to its nearest neighbor.
    """

    # input validation
    assert (
        coordinate.shape[-1] == 3 or coordinate.shape[-1] == 2
    ), "Each coordinate should have 3 values x, y, and z, or 2 values theta and phi"

    # pre-processing
    coordinate_flat = (
        coordinate.reshape(-1, 3)
        if coordinate.shape[-1] == 3
        else coordinate.reshape(-1, 2)
    )
    coordinate_cartesian = (
        coordinate_flat
        if coordinate.shape[-1] == 3
        else polar2cartesian(coordinate_flat)
    )

    _, distances = nearest_point(coordinate_cartesian, coordinate_cartesian, 2)

    return distances[:, 1]


@multimethod
@torch.no_grad()
@torch._dynamo.disable
def nearest_neighbor_distance(coordinate: torch.Tensor) -> torch.Tensor:
    """Return the angular distance to the nearest distinct neighbor for each point (torch).

    Distances are returned in radians and computed in Cartesian space via FAISS.
    Decorated with ``@torch.no_grad()`` and ``@torch._dynamo.disable`` so it is
    safe to call during compilation or inference.

    Args:
        coordinate (torch.Tensor): Point cloud given as unit Cartesian vectors of
            shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.

    Returns:
        torch.Tensor: 1-D tensor of shape ``(N,)`` containing the angular distance
            in radians from each point to its nearest neighbor.
    """

    # input validation
    assert (
        coordinate.shape[-1] == 3 or coordinate.shape[-1] == 2
    ), "Each coordinate should have 3 values x, y, and z, or 2 values theta and phi"

    # pre-processing
    coordinate_flat = (
        coordinate.reshape(-1, 3)
        if coordinate.shape[-1] == 3
        else coordinate.reshape(-1, 2)
    )
    coordinate_cartesian = copy_or_clone(
        (
            coordinate_flat
            if coordinate.shape[-1] == 3
            else polar2cartesian(coordinate_flat)
        )
    )

    _, distances = nearest_point(coordinate_cartesian, coordinate_cartesian, 2)

    return distances[:, 1]


@multimethod
def average_nn_angular_distance(coordinate: np.ndarray) -> float:  # type: ignore
    """Return the mean angular distance to the nearest distinct neighbor (numpy).

    Args:
        coordinate (np.ndarray): Point cloud given as unit Cartesian vectors of
            shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.

    Returns:
        float: Mean nearest-neighbor angular distance in radians.
    """
    return nearest_neighbor_distance(coordinate).mean()


@multimethod
@torch.no_grad()
@torch._dynamo.disable
# @torch.compiler.disable(recursive=False)
def average_nn_angular_distance(coordinate: torch.Tensor) -> float:
    """Return the mean angular distance to the nearest distinct neighbor (torch).

    Decorated with ``@torch.no_grad()`` and ``@torch._dynamo.disable`` so it is
    safe to call during compilation or inference.

    Args:
        coordinate (torch.Tensor): Point cloud given as unit Cartesian vectors of
            shape ``(..., 3)`` or polar coordinates of shape ``(..., 2)``.

    Returns:
        float: Mean nearest-neighbor angular distance in radians.
    """
    return nearest_neighbor_distance(coordinate).mean().item()


@multimethod
def colorize_locations(locations: np.ndarray, colormap: str = "plasma") -> np.ndarray:  # type: ignore
    """Assign a colormap color to each location based on its sequential index (numpy).

    Colors are drawn from a matplotlib colormap evaluated at uniformly spaced
    positions in ``[0, 1]``, then scaled to ``[0, 255]``. The result is useful
    for visualizing point ordering on the sphere.

    Args:
        locations (np.ndarray): Array of shape ``(..., 2)`` for polar coordinates
            or ``(..., 3)`` for Cartesian vectors. Only ``locations.shape[0]`` is
            used to determine the number of colors.
        colormap (str, optional): Name of a matplotlib colormap. Defaults to
            ``"plasma"``.

    Returns:
        np.ndarray: Float32 array of shape ``(..., 3)`` containing RGB values
            in the range ``[0, 255]``.
    """
    # Plasma color map to visualize index sequence
    normalized_indices = np.linspace(0, 1, locations.shape[0], dtype=np.float32)
    colors = cm.get_cmap(colormap)(normalized_indices)[:, :3] * 255

    return colors.reshape((*locations.shape[:-1], 3)).astype(np.float32, copy=False)


@multimethod
def colorize_locations(
    locations: torch.Tensor, colormap: str = "plasma"
) -> torch.Tensor:
    """Assign a colormap color to each location based on its sequential index (torch).

    Colors are drawn from a matplotlib colormap evaluated at uniformly spaced
    positions in ``[0, 1]``, then scaled to ``[0, 255]``. The colormap is
    evaluated on CPU (via numpy) and the result is moved to the same device as
    ``locations``.

    Args:
        locations (torch.Tensor): Tensor of shape ``(..., 2)`` for polar
            coordinates or ``(..., 3)`` for Cartesian vectors. Only
            ``locations.shape[0]`` is used to determine the number of colors.
        colormap (str, optional): Name of a matplotlib colormap. Defaults to
            ``"plasma"``.

    Returns:
        torch.Tensor: Float32 tensor of shape ``(..., 3)`` containing RGB values
            in the range ``[0, 255]``, on the same device as ``locations``.
    """
    # Plasma color map to visualize index sequence
    normalized_indices = torch.linspace(
        0,
        1,
        locations.shape[0],
        dtype=locations.dtype,
        device=locations.device,
    )
    cmap_in = normalized_indices.detach().cpu().numpy()
    colors = to_torch(
        cm.get_cmap(colormap)(cmap_in)[:, :3] * 255,
        dtype=torch.float32,
        device=locations.device,
    )

    return colors.view((*locations.shape[:-1], 3))


@multimethod
@torch._dynamo.disable
def average_pixel_area(coordinates: np.ndarray) -> float:  # type: ignore
    """Compute the average pixel area on the unit sphere for the given point cloud (numpy).

    Uses a Spherical Voronoi tessellation (requires float64 for numerical
    stability). Outlier Voronoi cells are removed via an IQR filter before
    averaging.

    Args:
        coordinates (np.ndarray): Point cloud given as polar coordinates
            ``(theta, phi)`` of shape ``(..., 2)`` or unit Cartesian vectors
            ``(x, y, z)`` of shape ``(..., 3)``.

    Returns:
        float: Mean inlier Voronoi cell area on the unit sphere (steradians).
    """

    # input validation
    assert (
        coordinates.shape[-1] == 2 or coordinates.shape[-1] == 3
    ), "Each coordinate must be in cartesian (x, y, z) or spherical (theta, phi) format"

    # Convert polar coordinates to cartesian coordinates, in pure function style
    _coordinates = np.ascontiguousarray(
        (
            polar2cartesian(coordinates)
            if coordinates.shape[-1] == 2
            else coordinates / np.linalg.norm(coordinates, axis=-1, keepdims=True)
        ).reshape(-1, 3),
        dtype=np.float64,
    )

    eps = 1e-6
    coords_q = np.round(_coordinates / eps) * eps
    coords_u = np.unique(coords_q, axis=0)
    coords_u /= np.linalg.norm(coords_u, axis=1, keepdims=True)

    sv = SphericalVoronoi(coords_u, threshold=1e-10)
    areas = np.array(sv.calculate_areas())

    Q1 = np.percentile(areas, 25)
    Q3 = np.percentile(areas, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    inlier_areas = areas[areas < upper_bound]

    return inlier_areas.mean(dtype=coordinates.dtype).item()


@multimethod
@torch.no_grad()
@torch._dynamo.disable
def average_pixel_area(coordinates: torch.Tensor) -> float:
    """Compute the average pixel area on the unit sphere for the given point cloud (torch).

    Converts inputs to numpy float64 internally for the Spherical Voronoi
    tessellation (scipy). Outlier Voronoi cells are removed via an IQR filter
    before averaging.

    Args:
        coordinates (torch.Tensor): Point cloud given as polar coordinates
            ``(theta, phi)`` of shape ``(..., 2)`` or unit Cartesian vectors
            ``(x, y, z)`` of shape ``(..., 3)``.

    Returns:
        float: Mean inlier Voronoi cell area on the unit sphere (steradians).
    """

    # input validation
    assert (
        coordinates.shape[-1] == 2 or coordinates.shape[-1] == 3
    ), "Each coordinate must be in cartesian (x, y, z) or spherical (theta, phi) format"

    # Convert polar coordinates to cartesian coordinates, in pure function style
    _coordinates = np.ascontiguousarray(
        to_numpy(
            (
                polar2cartesian(coordinates.clone())
                if coordinates.shape[-1] == 2
                else coordinates.clone()
                / torch.linalg.norm(coordinates, dim=-1, keepdims=True)
            )
        ).reshape(-1, 3),
        dtype=np.float64,
    )

    eps = 1e-6
    coords_q = np.round(_coordinates / eps) * eps
    coords_u = np.unique(coords_q, axis=0)
    coords_u /= np.linalg.norm(coords_u, axis=1, keepdims=True)

    sv = SphericalVoronoi(coords_u, threshold=1e-10)
    areas = torch.Tensor(sv.calculate_areas())

    Q1 = torch.quantile(areas, 0.25)
    Q3 = torch.quantile(areas, 0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    inlier_areas = areas[areas < upper_bound]

    return inlier_areas.mean(dtype=coordinates.dtype).item()
