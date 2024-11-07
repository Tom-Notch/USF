#!/usr/bin/env python3
#
# Created on Mon Nov 11 2024 17:15:17
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
"""Generate per-pixel unit-vector ("lens normal") maps for various camera models.

Supports pinhole, equidistant fisheye, equirectangular, and stereographic
projections. CLI entry point installed as ``generate_lens_normal_map``::

    generate_lens_normal_map -c path/to/camera_config.yaml
"""

import argparse
import os
import os.path as osp
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

from usf.utils import params
from usf.utils.files import parse_path, read_file
from usf.utils.spherical import polar2cartesian

# rotation matrix from camera frame to body frame
R_body_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

CURRENT_DIR = osp.dirname(osp.realpath(__file__))


def undistortion(ray_vectors: np.ndarray, distortion_parameters: tuple) -> np.ndarray:
    """undistortion:
       https://en.wikipedia.org/wiki/Distortion_%28optics%29#Software_correction
       https://github.com/ethz-asl/kalibr/wiki/supported-models
       https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a

    Args:
        ray_vectors (np.ndarray): z normalized ray vectors w/ shape (N, 3), will be reshaped regardless
        distortion_parameters (tuple): has shape (k1, k2, p1, p2) where k1, k2 are radial distortion parameters and p1, p2 are tangential distortion parameters

    Returns:
        np.ndarray: _description_
    """
    assert (
        len(distortion_parameters) == 4
    ), "distortion_parameters should have exactly 4 parameters: (k1, k2, p1, p2)"

    k1, k2, p1, p2 = distortion_parameters
    ray_vectors_original_shape = ray_vectors.shape
    ray_vectors = ray_vectors.reshape(-1, 3)
    rho_squared = np.square(
        np.linalg.norm(ray_vectors[:, :2], axis=1)
    )  # distance to (0, 0, 1) squared
    rho_quad = np.square(rho_squared)  # distance to (0, 0, 1) to the power of 4
    xy = ray_vectors[:, 0] * ray_vectors[:, 1]  # x * y
    dx = (
        ray_vectors[:, 0] * (k1 * rho_squared + k2 * rho_quad)
        + 2 * p1 * xy
        + p2 * (rho_squared + 2 * np.square(ray_vectors[:, 0]))
    )
    dy = (
        ray_vectors[:, 1] * (k1 * rho_squared + k2 * rho_quad)
        + 2 * p2 * xy
        + p1 * (rho_squared + 2 * np.square(ray_vectors[:, 1]))
    )
    ray_vectors[:, 0] = ray_vectors[:, 0] + dx
    ray_vectors[:, 1] = ray_vectors[:, 1] + dy

    return ray_vectors.reshape(
        ray_vectors_original_shape
    )  # actually inplace operation, but return for convenience


def calculate_pixel_ray(config: dict) -> np.ndarray:
    """calculate the ray vectors for each pixel in the image considering distortion, intrinsic and extrinsic parameters

    Args:
        config (dict): the config dict for the camera, must contain the following keys:
                        - image_width
                        - image_height
                        - intrinsic

    Returns:
        np.ndarray: per-pixel camera ray vectors in the camera frame w/ shape: (image_height, image_width, 3), dtype float32, each ray vector (in open3d.scene.cast_rays() format):
                    [direction_x, direction_y, direction_z]
                    direction vector will have unit length
    """
    assert "image_width" in config.keys(), "image_width is not specified in config"
    assert "image_height" in config.keys(), "image_height is not specified in config"
    assert "intrinsic" in config.keys(), "intrinsic is not specified in config"
    assert (
        "projection_parameters" in config["intrinsic"].keys()
    ), "projection_parameters is not specified in config intrinsic"
    assert config["intrinsic"]["model_type"] in params.CAMERA_MODEL_TYPE.keys(), (
        "camera model "
        + config["intrinsic"]["model_type"]
        + " has not been implemented yet!"
    )

    width, height = int(config["image_width"]), int(config["image_height"])

    # construct 2D pixel homogeneous coordinates (float32 end-to-end)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    homogeneous_pixel_vectors = np.stack(
        (grid_x, grid_y, np.ones(grid_x.shape, dtype=np.float32)), axis=-1
    )

    # Un-project the pixel vectors and normalize z to get the ray vectors
    if params.CAMERA_MODEL_TYPE[config["intrinsic"]["model_type"]] == "Pinhole":
        intrinsic_matrix = np.array(
            [
                [
                    config["intrinsic"]["projection_parameters"]["fx"],
                    0,
                    config["intrinsic"]["projection_parameters"]["cx"],
                ],
                [
                    0,
                    config["intrinsic"]["projection_parameters"]["fy"],
                    config["intrinsic"]["projection_parameters"]["cy"],
                ],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        ray_vectors = (
            np.linalg.inv(intrinsic_matrix)
            @ homogeneous_pixel_vectors.reshape(width * height, 3).T
        ).T  # flatten to shape (height * weight, 3) and un-project to z normalized plane with inverted intrinsic matrix

        if "distortion_parameters" in config["intrinsic"].keys():
            k1, k2, p1, p2 = (
                config["intrinsic"]["distortion_parameters"]["k1"],  # radial distortion
                config["intrinsic"]["distortion_parameters"]["k2"],  # radial distortion
                config["intrinsic"]["distortion_parameters"][
                    "p1"
                ],  # tangential distortion
                config["intrinsic"]["distortion_parameters"][
                    "p2"
                ],  # tangential distortion
            )
            ray_vectors = undistortion(ray_vectors, (k1, k2, p1, p2))

    elif params.CAMERA_MODEL_TYPE[config["intrinsic"]["model_type"]] == "Mei":
        # raise NotImplementedError("Mei camera model has not been implemented yet!")
        intrinsic_matrix = np.array(
            [
                [
                    config["intrinsic"]["projection_parameters"]["gamma1"],
                    0,
                    config["intrinsic"]["projection_parameters"]["u0"],
                ],
                [
                    0,
                    config["intrinsic"]["projection_parameters"]["gamma2"],
                    config["intrinsic"]["projection_parameters"]["v0"],
                ],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        ray_vectors = (
            np.linalg.inv(intrinsic_matrix)
            @ homogeneous_pixel_vectors.reshape(width * height, 3).T
        ).T  # flatten to shape (height * weight, 3) and un-project to z normalized plane with inverted intrinsic matrix

        if "distortion_parameters" in config["intrinsic"].keys():
            k1, k2, p1, p2 = (
                config["intrinsic"]["distortion_parameters"]["k1"],  # radial distortion
                config["intrinsic"]["distortion_parameters"]["k2"],  # radial distortion
                config["intrinsic"]["distortion_parameters"][
                    "p1"
                ],  # tangential distortion
                config["intrinsic"]["distortion_parameters"][
                    "p2"
                ],  # tangential distortion
            )
            ray_vectors = undistortion(ray_vectors, (k1, k2, p1, p2))

        # translate to camera frame
        # ! assume mirror_parameter Xi != 0.0
        Xi = config["intrinsic"]["mirror_parameters"]["xi"]
        Xi_squared = np.square(Xi)  # mirror parameter squared
        x_squared = np.square(ray_vectors[:, 0])  # z normalized x coordinate squared
        y_squared = np.square(ray_vectors[:, 1])  # z normalized y coordinate squared
        X_lambda = (Xi + np.sqrt(1 + (1 - Xi_squared) * (x_squared + y_squared))) / (
            x_squared + y_squared + 1
        )  # Single View Point Omnidirectional Camera Calibration from Planar Grids: https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
        ray_vectors = np.stack(
            (X_lambda * ray_vectors[:, 0], X_lambda * ray_vectors[:, 1], X_lambda - Xi),
            axis=1,
        )  # Single View Point Omnidirectional Camera Calibration from Planar Grids: https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
    else:  # e.g. unparameterized camera model where a ray direction is stored for each pixel
        raise NotImplementedError(
            "camera model "
            + config["intrinsic"]["model_type"]
            + " has not been implemented yet!"
        )

    # Rotate the ray vectors to the body frame
    ray_vectors = (R_body_camera @ ray_vectors.T).T

    ray_vectors = (
        ray_vectors / np.linalg.norm(ray_vectors, axis=1)[:, np.newaxis]
    )  # normalize depth to 1

    return ray_vectors.astype(np.float32, copy=False).reshape(height, width, -1)


def generate_lens_normal_map(
    pixel_vectors: np.ndarray, mask: np.ndarray, output_prefix: str
) -> None:
    """Save lens normal map as ``.npy`` and a visualization as ``.pdf``.

    Args:
        pixel_vectors (np.ndarray): Per-pixel direction vectors of shape (H, W, 3).
        mask (np.ndarray): Boolean mask of shape (H, W) indicating valid pixels.
        output_prefix (str): File path prefix; outputs are ``{prefix}_lens_normal_map.npy``
            and ``{prefix}_lens_normal_map.pdf``.
    """
    camera_name = osp.basename(output_prefix)

    # Calculate the relative directions
    relative_directions = pixel_vectors.copy()

    # Normalize directions to unit vectors
    norms = np.linalg.norm(relative_directions, axis=-1, keepdims=True)
    normalized_directions = relative_directions / norms

    np.save(f"{output_prefix}_lens_normal_map.npy", normalized_directions)

    # Map normalized directions to RGB color values in the CG style (X -> Red, Y -> Green, Z -> Blue)
    # Shift range from [-1, 1] to [0, 1] for display purposes
    lens_normal_map = ((normalized_directions + 1) / 2 * 255).astype(np.uint8)

    # Apply the mask
    lens_normal_map[~(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) > 0)] = (
        0  # Ignore background
    )

    output_image = f"{output_prefix}_lens_normal_map.pdf"

    # Display the normal map in CG style colors
    plt.figure(figsize=(8, 8))
    plt.imshow(lens_normal_map)
    plt.title(f"Lens Normal Map of {camera_name}")
    plt.axis("off")
    plt.savefig(output_image, transparent=True, bbox_inches="tight", pad_inches=0)
    print(f"lens normal map saved as {output_image}")
    plt.show()


def generate_pinhole_lens_normal_map(
    horizontal_fov: float, vertical_fov: float, height: int, width: int
) -> np.ndarray:
    """
    Generate a lens normal map for a pinhole camera given horizontal and vertical field of view.
    This version returns a map in which the top of the image is up in the camera coordinate system.

    Assumes the following coordinate system:
      - x: forward (out of the camera)
      - y: left
      - z: up

    Args:
        horizontal_fov (float): Horizontal field of view in radians.
        vertical_fov (float): Vertical field of view in radians.
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        np.ndarray: lens normal map of shape (height, width, 3), dtype float32, where each (x, y, z) is a unit vector.
    """
    # Compute focal lengths from FOV (f = (sensor_width/2) / tan(fov/2)).
    # Here we use image width and height as surrogates for sensor size.
    w = np.float32(width)
    h = np.float32(height)
    fx = (w / 2) / np.tan(np.float32(horizontal_fov) / 2)
    fy = (h / 2) / np.tan(np.float32(vertical_fov) / 2)
    cx = w / 2
    cy = h / 2

    # Change sign of fy to flip the vertical axis,
    # so that the camera coordinate system has y upward.
    K = np.array([[fx, 0, cx], [0, -fy, cy], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    # Create a grid of pixel coordinates.
    u_coords = np.arange(width, dtype=np.float32)  # pixel x-coordinates: 0 to width-1.
    v_coords = np.arange(
        height, dtype=np.float32
    )  # pixel y-coordinates: 0 to height-1.
    u, v = np.meshgrid(u_coords, v_coords)
    ones = np.ones_like(u)
    # Each pixel is represented in homogeneous coordinates: (u, v, 1)
    homogeneous_pixels = np.stack([u, v, ones], axis=-1)  # shape (height, width, 3)

    # Flatten the pixel grid so that we can apply the inverse intrinsic matrix.
    pixels_flat = homogeneous_pixels.reshape(-1, 3).T  # shape (3, height x width)
    # Unproject pixels: each column is a 3D ray in the camera coordinate system.
    rays = K_inv @ pixels_flat  # shape (3, height x width)
    rays = rays.T  # shape (height x width, 3)

    # Transform from standard camera coordinates (x: right, y: up, z: forward)
    # to our desired coordinates (x: forward, y: left, z: up).
    # We want:
    #   new_x = z (forward)
    #   new_y = - x (left; flipping right)
    #   new_z = y (up)
    T = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)

    # Apply the transformation T to each ray.
    rays_transformed = (T @ rays.T).T.reshape(height, width, 3)

    # Normalize again just to be safe.
    norm = np.linalg.norm(rays_transformed, axis=-1, keepdims=True)
    lens_normal_map = rays_transformed / norm

    return lens_normal_map.astype(np.float32, copy=False)


def generate_fisheye_lens_normal_map(fov: float, height: int, width: int):
    """
    Generate a lens normal map for a fisheye camera using equidistant projection.

    Args:
        fov (float): Full diagonal field of view in radians (e.g., np.pi for 180° fisheye).
        height (int): Image height in pixels.
        width (int): Image width in pixels.

    Returns:
        normal_map (np.ndarray): shape (height, width, 3), dtype float32, each pixel is a unit ray in camera frame.
        mask (np.ndarray): shape (height, width), bool array indicating valid pixels (within FOV).
    """
    # Coordinate grid centered at image center (float32 end-to-end)
    w = np.float32(width)
    h = np.float32(height)
    cx = w / 2
    cy = h / 2
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x = (u - cx) / (w / 2)  # normalized to [-1, 1]
    y = -(v - cy) / (h / 2)  # inverted y to match camera coordinate system

    r = np.sqrt(x**2 + y**2)

    # Equidistant mapping: theta = r * f, f is chosen so that r=1 maps to theta = fov/2
    fov_f = np.float32(fov)
    half_fov = fov_f / 2
    theta = r * half_fov

    # Mask: only use directions within FOV
    mask = theta <= half_fov

    # Direction vector in spherical coords
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # azimuthal angle φ from x and y (note: atan2 uses y first)
    phi = np.arctan2(y, x)

    # Convert to 3D ray: spherical to Cartesian
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = cos_theta

    rays = np.stack([dx, dy, dz], axis=-1)  # (height, width, 3)

    # Transform to your convention:
    # x_cam = dz (forward), y_cam = -dx (left), z_cam = dy (up)
    T = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)
    rays_transformed = rays @ T.T

    # Normalize rays (just in case)
    norm = np.linalg.norm(rays_transformed, axis=-1, keepdims=True)
    normal_map = rays_transformed / np.clip(norm, a_min=np.float32(1e-8), a_max=None)

    return normal_map.astype(np.float32, copy=False), mask.astype(bool)


def generate_equirectangular_normal_map(height: int, width: int) -> np.ndarray:
    """
    Generate a lens_normal_map for an equirectangular panorama of shape (height, width).

    Assumes:
      - row i in [0..height-1] => latitude theta in [-pi/2..+pi/2]
      - col j in [0..width-1]  => longitude phi in [-pi..+pi]

    Args:
        height (int): Number of rows (pixels) in the image.
        width (int): Number of columns (pixels) in the image.

    Returns:
      lens_normal_map of shape (height, width, 3), dtype float32, with each (x,y,z)
      a unit vector on the sphere.
    """

    # 1) Define the latitude (theta) range: [-pi/2..+pi/2] (non-inclusive)
    theta_values = np.linspace(
        -np.pi / 2, np.pi / 2, height + 1, endpoint=False, dtype=np.float32
    )[1:]
    theta_values = theta_values[
        ::-1
    ]  # Reverse the order of theta_values to match the image pixel arrangement

    # 2) Define the longitude (phi) range: [-pi..+pi] (non-overlapping)
    phi_values = np.linspace(-np.pi, np.pi, width, endpoint=False, dtype=np.float32)
    phi_values = phi_values[
        ::-1
    ]  # Reverse the order of phi_values to match the image pixel arrangement

    # 3) Create a 2D grid (theta_grid, phi_grid) with shape (height, width)
    #    indexing='ij' => first dimension (i) indexes rows => theta, second (j) => phi
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing="ij")
    # After meshgrid, theta_grid.shape => (height, width), same for phi_grid

    # 4) Compute (x, y, z)
    polar = np.dstack([theta_grid, phi_grid])
    lens_normal_map = polar2cartesian(polar)

    return lens_normal_map.astype(np.float32, copy=False)


def generate_stereographic_normal_map(
    height: int,
    width: int,
    projection_point: tuple | list | np.ndarray = (-1, 0, 0),
    up: tuple | list | np.ndarray = (0, 0, 1),
) -> np.ndarray:
    """
    Generate a lens normal map for a stereographic projection of shape (height, width, 3).

    The stereographic projection is computed as if an image were placed in front of the unit sphere.
    The projection is done with respect to a configurable projection point (default (-1, 0, 0)),
    so that the tangent plane is defined at the antipodal point T = -projection_point. The horizontal
    coordinate is flipped to account for a camera looking from the projection center toward the image,
    and the vertical coordinate is flipped so that the top of the image maps to a positive direction.

    Assumes:
      - Rows i in [0..height-1] and columns j in [0..width-1] index the image.
      - u = (width/2 - j) and v = (height/2 - i) are computed to flip left/right and top/bottom, respectively.
      - The coordinates are normalized by an isotropic scale factor = max(height, width)/2.

    Args:
        height (int): Number of rows (pixels) in the image.
        width (int): Number of columns (pixels) in the image.
        projection_point (tuple | list | np.ndarray, optional): A 3-element sequence specifying the projection point
                                                               on the unit sphere. Defaults to (-1, 0, 0).
        up (tuple | list | np.ndarray, optional): A 3-element sequence specifying the desired “up” direction. Defaults to (0, 0, 1).

    Returns:
        np.ndarray: A lens normal map of shape (height, width, 3), dtype float32, where each (x, y, z) is a unit vector on the sphere.

    Note:
        A one-time warning is issued if height and width differ since a non-square domain may lead to non-uniform distortions.
    """
    if height != width:
        warnings.warn(
            "The input dimensions are not square. Stereographic projection may lead to non-uniform distortions.",
            UserWarning,
            stacklevel=2,
        )

    # 1) Create a grid of pixel indices.
    ys, xs = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    # 2) Map pixel indices to plane coordinates.
    # Flip horizontal: u = (width/2 - x)
    # Flip vertical:   v = (height/2 - y)
    u = np.float32(width) / 2 - xs
    v = np.float32(height) / 2 - ys

    # 3) Normalize by an isotropic scale (max(height, width)/2).
    scale = np.float32(max(height, width)) / 2
    u = u / scale
    v = v / scale

    # 4) Normalize the projection point P and compute the tangent point T = -P.
    P = np.array(projection_point, dtype=np.float32)
    P = P / np.linalg.norm(P)
    T = -P  # This is where the center of the image maps.

    # 5) Compute an orthonormal basis for the tangent plane at T.
    #    e2: “up” basis is obtained by projecting the desired up vector onto the tangent plane.
    up_vector = np.array(up, dtype=np.float32)
    up_vector = up_vector / np.linalg.norm(up_vector)
    e2 = up_vector - np.dot(up_vector, T) * T
    if np.linalg.norm(e2) < 1e-8:
        raise ValueError("The chosen up vector is colinear with the tangent direction.")
    e2 = e2 / np.linalg.norm(e2)
    #    e1: horizontal basis is given by the cross product of e2 and T.
    e1 = np.cross(e2, T)
    e1 = e1 / np.linalg.norm(e1)

    # 6) Compute the scaling factor t for each pixel.
    t = np.float32(4) / (np.float32(4) + u**2 + v**2)

    # 7) Compute the polar coordinates for each pixel.
    #     When u=v=0, t=1 and the result is T.
    S = (2 * t - 1)[..., None] * T + t[..., None] * (
        u[..., None] * e1 + v[..., None] * e2
    )

    # S has shape (height, width, 3) and contains unit vectors on the sphere.
    return S.astype(np.float32, copy=False)


def main() -> None:
    """Parse CLI args, read camera config, generate and save the lens normal map."""
    parser = argparse.ArgumentParser(
        description="Generate lens normal map for calibrated camera with intrinsic and distortion parameters."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="path to the camera config file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="",
        type=str,
        help="folder to the output of lens normal map file",
    )
    args = parser.parse_args()

    # Read the camera config file, must contain the following keys:
    # - image_width
    # - image_height
    # - intrinsic
    config_path = parse_path(args.config)
    config_dir = osp.dirname(config_path)
    camera_name = osp.splitext(osp.basename(config_path))[0]
    config = read_file(config_path)
    print("camera config:")
    print(config)

    # determine if output is a path or a file
    if args.output == "":
        os.makedirs(osp.join(config_dir, "lens_normal_maps"), exist_ok=True)
        output_prefix = osp.join(config_dir, "lens_normal_maps", f"{camera_name}")
    elif osp.isdir(args.output):
        # make sure the output path is a valid directory, create if not exist
        if not osp.exists(args.output):
            os.makedirs(args.output)
        output_prefix = osp.join(args.output, f"{camera_name}")
    else:
        raise ValueError("Invalid output directory")

    height, width = int(config["image_height"]), int(config["image_width"])

    # Calculate the pixel ray vectors
    ray_vectors = calculate_pixel_ray(config)

    if "mask" in config.keys():
        # NOTE: procedures:
        # 1. resize mask to image size
        # 2. binarize the mask
        _, binarized_mask = cv2.threshold(
            cv2.resize(config["mask"], (width, height)), 128, 255, cv2.THRESH_BINARY
        )
        mask = binarized_mask
    else:
        mask = (
            np.ones((height, width, 3), dtype=np.uint8) * 255
        )  # NOTE: if no mask is provided, use a white mask
        # 255 is white, 0 is black

    generate_lens_normal_map(ray_vectors, mask, output_prefix)


if __name__ == "__main__":
    main()
