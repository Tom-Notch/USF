#!/usr/bin/env python3
#
# Created on Tue Feb 11 2025 16:57:53
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import patches
from multimethod import multimethod
from PIL import Image
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import functional as F

from usf.dataset.iou import pairwiseIoU
from usf.generate_lens_normal_map import generate_equirectangular_normal_map
from usf.network.layer.spherical.circle_pool import CirclePool
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.cache import ephemeral_cache
from usf.utils.files import parse_path, read_file
from usf.utils.nearest_neighbor import nearest_point
from usf.utils.spherical import (
    azim_elev_to_rotation_matrix,
    cartesian2polar,
    compute_inside_fov_mask,
    nearest_neighbor_distance,
    normalize_polar,
    polar2cartesian,
    sample_random_rotation_matrices,
    spherical_distance,
)
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import (
    copy_or_clone,
    fixed_seed,
    string_to_seed,
    to_numpy,
    to_torch,
)


class PandoraDataset(Dataset):
    """PANDORA spherical object detection dataset.

    Each sample is a 360° equirectangular image annotated with rotated bounding
    field-of-view (RBFoV) labels. Images are loaded, optionally downsampled,
    projected onto the sphere, and augmented with random rotation, color jitter,
    and horizontal flip. An optional ``output_vector`` allows resampling onto a
    different spherical grid (e.g. pinhole or fisheye lens normal map).

    The ``@`` operator (``n @ dataset``) sub-samples the dataset.
    """

    def __init__(
        self,
        dataset_base_path: str,
        dataset_type: str,
        augmentation: dict[str, float] | None = None,
        downsample_image_size: tuple[int, int] = (960, 480),
        output_vector: torch.Tensor | np.ndarray | None = None,
        output_vector_mask: torch.Tensor | np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        seed: str | None = None,
    ):
        """
        Initializes the PANDORA dataset

        Args:
            dataset_base_path (str): Base directory where the PANDORA dataset is stored
            dataset_type (str): 'train' or 'test', specifying which dataset to load
            augmentation (dict[str, float] | None, optional): Dictionary of data augmentation probabilities. Defaults to None.
            downsample_image_size (tuple[int, int], optional): output shape of image in (width, height). Defaults to (960, 480).
            output_vector (torch.Tensor | np.ndarray | None, optional): Optional forward-facing lens normal map, usually pinhole or fisheye. Defaults to None.
            output_vector_mask (torch.Tensor | np.ndarray | None, optional): Optional mask for the output vector, if provided. Must be provided if fisheye. Defaults to None.
            meta (dict[str, Any] | None, optional): Optional meta dictionary, e.g. value mean/std for normalization, class weights, etc. Defaults to None.
            seed (str | None, optional): Random seed for reproducibility. Defaults to None.
        """
        self.downsample_image_size = downsample_image_size
        self.set_output_vector(output_vector)
        self.set_output_vector_mask(output_vector_mask)

        # --- Data Augmentation ---
        augmentation = augmentation or {}  # default to empty dict

        # value space augmentation
        value_augmentations = []
        value_augmentations.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.0,
                        contrast=0.0,
                        saturation=0.4,
                        hue=0.1,
                    )
                ],
                p=augmentation.get("chroma_jitter", 0.0),
            )
        )
        value_augmentations.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.0,
                        hue=0.0,
                    )
                ],
                p=augmentation.get("luma_jitter", 0.0),
            )
        )
        value_augmentations.append(
            T.RandomApply(
                [
                    T.GaussianBlur(kernel_size=(15, 15), sigma=(0.8, 2.0)),
                ],  # small objects may be blurred out, so reduce sigma
                p=augmentation.get("gaussian_blur", 0.0),
            )
        )
        value_augmentations.append(
            T.RandomApply(
                [
                    T.Grayscale(num_output_channels=3),
                ],
                p=augmentation.get("gray_scale", 0.0),
            )
        )
        # independent Bernoulli for each augmentation, order matters here because grayscale kills color
        self.value_augmenter = T.Compose(value_augmentations)

        # geometric augmentation
        # reflection
        self.random_horizontal_reflection_probability = augmentation.get(
            "horizontal_reflection", 0.0
        )
        self.random_vertical_reflection_probability = augmentation.get(
            "vertical_reflection", 0.0
        )  # this is unnecessary with random rotation

        # erase
        self.random_erase_probability = augmentation.get("erase", 0.0)

        # rotation
        self.random_rotation_probability = augmentation.get("rotation", 0.0)
        if (
            self.random_rotation_probability > 0.0
            and getattr(self, "color_resampler", None) is None
        ):
            self.color_resampler = ValueSampler(
                {
                    "value_sampler": "nearest_neighbor",
                    "value_sampler_config": {"num_neighbors": 4},
                    "reject_oo_fov_value": True,
                }
            )
            self.label_resampler = ValueSampler(
                {
                    "value_sampler": "nearest_neighbor",
                    "value_sampler_config": {"num_neighbors": 1},
                    "reject_oo_fov_value": True,
                }
            )

        dataset_base_path = parse_path(dataset_base_path)
        assert dataset_base_path, "Invalid dataset_base_path"

        self.images_folder_path = parse_path(dataset_base_path + "/images")

        # Build file paths based on the dataset type.
        if dataset_type == "train":
            labels_path = parse_path(dataset_base_path + "/annotations/train.json")
        elif dataset_type == "test":
            labels_path = parse_path(dataset_base_path + "/annotations/test.json")
        else:
            raise ValueError(
                f"dataset_type {dataset_type} must be either 'train' or 'test'"
            )

        assert (
            self.images_folder_path and labels_path
        ), "PANDORA subfolder structure not following canonical"

        # parse labels
        raw_labels = read_file(labels_path)

        # record names for visualization
        self.category_name_table: dict[int, str] = {}
        for category in raw_labels["categories"]:
            self.category_name_table[int(category["id"]) - 1] = category["name"]

        # initialize category color map for visualization
        self.category_colormap = self._get_category_colormap(
            len(self.category_name_table)
        )

        # record bounding boxes and categories
        self.directory: dict[str, dict] = {}
        for raw_label in raw_labels["annotations"]:
            image_id = raw_label["image_id"]
            if image_id not in self.directory:
                entry: dict[str, Any] = {}
                raw_rbfov = np.empty((0, 6))
                converted_rbfov = np.empty((0, 6))
                self.directory[image_id] = entry
            else:
                entry = self.directory[image_id]
                raw_rbfov = entry["raw_rbfov"]
                converted_rbfov = entry["converted_rbfov"]

            # each row format: (RBFoV, category_id)
            raw_rbfov = np.vstack(
                (
                    raw_rbfov,
                    np.array(
                        [
                            *raw_label["bbox"],
                            raw_label["category_id"] - 1,  # raw ranged from 1 to 47
                        ]
                    ),
                )
            )

            converted_rbfov = np.vstack(
                (
                    converted_rbfov,
                    np.array(
                        [
                            *self._convert_convention(raw_label["bbox"]),
                            raw_label["category_id"] - 1,  # raw ranged from 1 to 47
                        ]
                    ),
                )
            )

            entry["raw_rbfov"] = raw_rbfov
            entry["converted_rbfov"] = converted_rbfov

        # record file names
        for image_metadata in raw_labels["images"]:
            self.directory[image_metadata["id"]]["file_name"] = image_metadata[
                "file_name"
            ]

        # convert to sorted list
        self.directory: list[dict] = [
            self.directory[key] for key in sorted(self.directory.keys())
        ]

        self.meta = meta or {}
        self.seed = seed

    def set_output_vector(
        self, output_vector: torch.Tensor | np.ndarray | None
    ) -> PandoraDataset:
        """Set or clear a fixed output grid for reprojection.

        When set, precomputes rotation matrices for cube-face sampling and
        initializes nearest-neighbor resamplers for color and label
        interpolation.

        Args:
            output_vector (torch.Tensor | np.ndarray | None): Unit vectors of
                shape (N, 3) defining the target grid, or None to clear.

        Returns:
            PandoraDataset: ``self`` for chaining.
        """
        if output_vector is not None:
            self.output_vector = to_torch(output_vector)

            # --- Precompute Rotations ---
            face_azs = [
                0,
                torch.pi / 2,
                torch.pi,
                3 * torch.pi / 2,
            ]  # 0°, 90°, 180°, 270°
            horizontal_faces = [(az, 0.0) for az in face_azs]
            up_down_faces = [(0.0, torch.pi / 2), (0.0, -torch.pi / 2)]
            directions = horizontal_faces + up_down_faces

            # corner_azs = [
            #     torch.pi / 4,
            #     3 * torch.pi / 4,
            #     5 * torch.pi / 4,
            #     7 * torch.pi / 4,
            # ]  # 45°,135°,225°,315°
            # corner_els = [torch.pi / 4, -torch.pi / 4]  # ±45°
            # corner_faces = [(az, el) for az in corner_azs for el in corner_els]

            # directions = horizontal_faces + up_down_faces + corner_faces

            # shape (14,2)
            azimuth_elevation = torch.tensor(directions, dtype=torch.float32)

            # batch-generate 14 rotation matrices: (14,3,3)
            self.rotations = azim_elev_to_rotation_matrix(azimuth_elevation)

            if getattr(self, "color_resampler", None) is None:
                self.color_resampler = ValueSampler(
                    {
                        "value_sampler": "nearest_neighbor",
                        "value_sampler_config": {"num_neighbors": 4},
                        "reject_oo_fov_value": True,
                    }
                )
                self.label_resampler = ValueSampler(
                    {
                        "value_sampler": "nearest_neighbor",
                        "value_sampler_config": {"num_neighbors": 1},
                        "reject_oo_fov_value": True,
                    }
                )
        else:
            if getattr(self, "output_vector", None) is not None:
                del self.output_vector
            if getattr(self, "rotations", None) is not None:
                del self.rotations

        return self

    def set_output_vector_mask(
        self, output_vector_mask: torch.Tensor | np.ndarray | None
    ) -> PandoraDataset:
        """Set or clear a validity mask for the output grid.

        Normalizes non-boolean masks to [0, 1] and thresholds at 0.5.

        Args:
            output_vector_mask (torch.Tensor | np.ndarray | None): Boolean or
                float mask of shape (N,), or None to clear.

        Returns:
            PandoraDataset: ``self`` for chaining.
        """
        if output_vector_mask is not None:
            self.output_vector_mask = to_torch(output_vector_mask)
            if self.output_vector_mask.dtype != torch.bool:  # if mask is not binary
                if self.output_vector_mask.max() > 1:  # if max > 1
                    self.output_vector_mask = (
                        self.output_vector_mask.float() / self.output_vector_mask.max()
                    )  # Normalize to [0, 1]

                self.output_vector_mask = self.output_vector_mask > 0.5  # binarize

            if self.output_vector is not None:
                assert (
                    self.output_vector_mask.shape == self.output_vector.shape[:2]
                ), "output_vector_mask and output_vector must have the same shape"
        else:
            if getattr(self, "output_vector_mask", None) is not None:
                del self.output_vector_mask

        return self

    def __rmatmul__(self, other: int | float) -> PandoraDataset:
        """Supports x @ PandoraDataset, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Raises:
            TypeError: If `other` is not an int or float.

        Returns:
            PandoraDataset
        """
        if isinstance(other, float):
            num_samples = int(other * len(self))
        elif isinstance(other, int):
            num_samples = min(other, len(self))
        else:
            raise TypeError(f"Left of @ must be int or float, got {type(other)}")

        num_samples = min(max(num_samples, 1), len(self))
        random_indices = np.random.choice(len(self), num_samples, replace=False)

        dataset_copy = deepcopy(self)
        dataset_copy.directory = [self.directory[i] for i in random_indices]

        return dataset_copy

    def _convert_convention(self, bbox: list) -> torch.Tensor:
        """From the paper:
            The raw RBFoV is defined by (θ, φ, α, β, γ)
            where θ and φ are the longitude and latitude coordinates of the object center
            and α, β denote the up-down and left-right field-of-view angles of the object’s occupation
            γ represents the angle (clockwise is positive, counterclockwise is negative) of the rotation
            of the tangent plane of the RBFoV along the axis O⃗ M (The M is the tangent point (θ, φ))
            The range of values of γ is [−90, 90].

            In my convention, polar is in (θ, φ), θ is latitude in [-pi/2, pi/2], φ is longitude in [-pi, pi]
            And (θ, φ) = (0, 0) corresponds to (1, 0, 0) in cartesian coordinate, details refer to docs/
            (α, β) both in [0, 2 x pi], γ in [-pi/2, pi/2]

            And because in equirectangular projection, we are looking out of the camera when we look at the image,
            the left and right are reversed when we visualize the projected spherical image because we are looking into the camera.
            As a result, we should negate longitude and γ

            Apparently sth's wrong with their up-down and left-right fov angle order, so we have to also reverse these 2
        Args:
            bbox (list): raw list label

        Returns:
            torch.Tensor: shape (5,)
        """
        bbox = copy_or_clone(torch.Tensor(bbox))
        bbox[2:5] = bbox[2:5] * torch.pi / 180  # Convert α, β, and γ to radians
        return torch.Tensor(
            [
                *normalize_polar(np.array([bbox[1], -bbox[0]])),
                bbox[3],
                bbox[2],
                self.normalize_rotation_differentiable(-bbox[4]),
            ]
        )  # revert θ and φ, α and β, and negate longitude and γ

    def _revert_convention(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Reverts the bbox from the normalized convention back to the original RBFoV format.

        Args:
            bbox (torch.Tensor): Converted bbox of shape (5,)

        Returns:
            List: Original bbox format with angles in degrees.
        """
        bbox = copy_or_clone(bbox)

        # Revert polar coordinate transformation (undo negation of longitude)
        polar = normalize_polar(bbox[:, :2])

        # Swap α and β back to their original order
        alpha_original = bbox[:, 3]  # Originally beta
        beta_original = bbox[:, 2]  # Originally alpha

        # Convert radians back to degrees
        alpha_original *= 180 / torch.pi
        beta_original *= 180 / torch.pi
        gamma_original = (
            self.normalize_rotation_differentiable(-bbox[:, 4]) * 180 / torch.pi
        )

        return torch.hstack(
            (
                -polar[:, 1],
                polar[:, 0],
                beta_original,
                alpha_original,
                gamma_original,
            )
        )

    @staticmethod
    def _get_category_colormap(num_categories: int, alpha=0.8) -> dict:
        cmap = plt.get_cmap("hsv", num_categories)
        colors = {i: cmap(i, alpha) for i in range(num_categories)}
        return colors

    @staticmethod
    def _equirectangular_projection(
        image: np.ndarray | torch.Tensor,
    ) -> SphericalImage:
        """
        Internal method to compute a SphericalImage from a given image using equirectangular projection.
        It calls on generate_equirectangular_normal_map to compute the per-pixel unit vectors.

        Args:
            image (np.ndarray | torch.Tensor): Input image with shape (H, W, C).

        Returns:
            SphericalImage: An instance containing:
                - value: flattened image data of shape (H*W, C)
                - vector: flattened lens normal map of shape (H*W, 3)
                - polar: left as None (to be computed in SphericalImage.__post_init__)
                - path: None
        """
        _image = to_torch(image)

        H, W, C = _image.shape
        # Warn if the _image W != 2 * H
        if W != 2 * H:
            warnings.warn(
                f"The input image has height {H} and width {W}. Equirectangular projection may lead to non-uniform distortions.",
                UserWarning,
                stacklevel=2,
            )

        # Use the imported function to generate the per-pixel unit vectors.
        vector = generate_equirectangular_normal_map(H, W).reshape(-1, 3)

        # Create and return the SphericalImage instance.
        return SphericalImage(
            value=_image.view(-1, C),
            vector=to_torch(vector),
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.directory)

    def _sample_images_from_random_direction(
        self,
        spherical_images: BatchSphericalImage,
        spherical_valid_pixel_mask: BatchSphericalImage,
        converted_rbfovs: torch.Tensor,
        random_direction: bool = True,
    ) -> tuple[BatchSphericalImage, BatchSphericalImage, torch.Tensor]:
        """Sample spherical images from 1/14 random directions using the precomputed rotation matrices.

        Args:
            spherical_images (BatchSphericalImage): Input batch spherical image.
            spherical_valid_pixel_mask (BatchSphericalImage): Valid pixel mask for the spherical images.
            converted_rbfovs (torch.Tensor): converted RBFoVs for the sample.
            random_direction (bool, optional): Whether to randomly select one of the 14 directions. Defaults to True.

        Returns:
            tuple[BatchSphericalImage, torch.Tensor]: sample batch spherical images
        """
        output_vector_flat = (
            self.output_vector.view(-1, 3)
            if getattr(self, "output_vector_mask", None) is None
            else self.output_vector[self.output_vector_mask].view(-1, 3)
        )

        if random_direction:
            rotation = self.rotations[
                torch.randint(0, self.rotations.shape[0], (1,)).item()
            ].to(
                dtype=output_vector_flat.dtype, device=output_vector_flat.device
            )  # randomly select one rotation matrix

            # Rotate the output_vector for each face
            face_vector = (rotation @ output_vector_flat.T).T  # shape (N, 3)

            # Inverse rotate the converted_rbfovs
            converted_rbfovs = self._rotate_converted_rbfovs(
                converted_rbfovs, rotation.T
            )  # Inverse rotation matrix
        else:
            # Use the output_vector as is
            face_vector = output_vector_flat

        sampled_spherical_images = self.color_resampler(
            spherical_images,
            face_vector,
        )
        sampled_spherical_valid_pixel_mask = self.label_resampler(
            spherical_valid_pixel_mask,
            face_vector,
        )

        # set the vector to output_vector_flat
        sampled_spherical_images.vector = output_vector_flat.to(
            sampled_spherical_images.batch_value.device
        )
        sampled_spherical_valid_pixel_mask.vector = sampled_spherical_images.vector

        return (
            sampled_spherical_images,
            sampled_spherical_valid_pixel_mask,
            converted_rbfovs,
        )

    @torch.inference_mode()
    def compute_rgb_mean_std(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-channel mean and standard deviation over the dataset (no augmentation),
        using only valid pixels (non-black borders) from RGB images.

        Args:
            stride (int, optional): process every `stride`-th image to speed up (1 = all images). Defaults to 1.

        Returns:
            (mean[3], std[3]) on [0,1] scale as np.float64 arrays.
        """
        sum_channel = np.zeros(3, dtype=np.float64)
        squared_sum_channel = np.zeros(3, dtype=np.float64)
        total_pixel_count = 0

        for image_index, entry in enumerate(self.directory):
            if (image_index % stride) != 0:
                continue

            rgb_path = self.images_folder_path + "/" + entry["file_name"]
            rgb_image = read_file(rgb_path)  # expected (H, W, 4) uint8
            if rgb_image.dtype != np.uint8:  # ensure uint8
                rgb_image = rgb_image.astype(np.uint8, copy=False)
            if rgb_image.shape[-1] >= 3:  # drop alpha if exists
                rgb_image = rgb_image[..., :3]
            else:
                raise ValueError(
                    f"Expected at least 3 channels in {rgb_path}, got {rgb_image.shape}"
                )

            rgb_image = rgb_image.reshape(-1, 3).astype(np.float64) / 255.0

            # Accumulate sums
            sum_channel += rgb_image.sum(axis=0)
            squared_sum_channel += (rgb_image * rgb_image).sum(axis=0)
            total_pixel_count += rgb_image.shape[0]

        if total_pixel_count == 0:
            # Degenerate case: no valid pixels found
            return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)

        mean_channel = sum_channel / total_pixel_count
        variance_channel = (
            squared_sum_channel / total_pixel_count - mean_channel * mean_channel
        )
        std_channel = np.sqrt(np.maximum(variance_channel, 1e-12))
        return mean_channel, std_channel

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary with keys:
                - "inputs":
                    - "images": a tensor of shape (1, H, W, C) created by stacking image tensors.
                    - "spherical_images": a BatchSphericalImage instance built from the list of spherical images.
                - "labels":
                    - "converted_rbfovs": List of list of converted rbfovs
                    # - "raw_rbfovs": List of list of raw rbfovs
                    why double list? to have consistent format be it a single sample or a batch
        """
        entry = self.directory[idx]

        # read image on the fly
        image = read_file(self.images_folder_path + "/" + entry["file_name"])

        # fetch label
        sample_converted_rbfov = to_torch(copy_or_clone(entry["converted_rbfov"]))

        # determine seed for reproducibility
        sample_seed = (
            None if self.seed is None else string_to_seed(self.seed + f" {idx}")
        )

        # perform value augmentation before resizing
        # draw all the rest of the augmentations in a single fixed_seed block
        with fixed_seed(sample_seed):
            image = np.array(self.value_augmenter(Image.fromarray(image)))
            random_horizontal_reflection = bool(
                (torch.rand(()) < self.random_horizontal_reflection_probability).item()
            )
            random_vertical_reflection = bool(
                (torch.rand(()) < self.random_vertical_reflection_probability).item()
            )
            random_erase = bool((torch.rand(()) < self.random_erase_probability).item())
            random_rotation = bool(
                (torch.rand(()) < self.random_rotation_probability).item()
            )

        # resize
        image = cv2.resize(
            image,
            self.downsample_image_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # apply possible random reflection
        if random_horizontal_reflection:
            image = np.ascontiguousarray(image[:, ::-1, :])
            sample_converted_rbfov[:, 1] = -sample_converted_rbfov[:, 1]  # negate φ
            sample_converted_rbfov[:, 4] = -sample_converted_rbfov[:, 4]  # negate γ

        if random_vertical_reflection:
            image = np.ascontiguousarray(image[::-1, :, :])
            sample_converted_rbfov[:, 0] = -sample_converted_rbfov[:, 0]  # negate θ
            sample_converted_rbfov[:, 4] = -sample_converted_rbfov[:, 4]  # negate γ

        image_tensor = to_torch(image, dtype=torch.float32).unsqueeze(
            0
        )  # shape (1, H, W, C)

        # apply possible random erase
        valid_pixel_mask_chw = torch.ones(
            (1, 1, *image_tensor.shape[1:3]), dtype=torch.bool
        )  # (1, 1, H, W), for normalization in preprocessing
        if random_erase:
            image_chw = image_tensor.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            vector = to_torch(
                generate_equirectangular_normal_map(
                    self.downsample_image_size[-1],
                    self.downsample_image_size[0],
                )
            )
            valid_vector_mask_chw = torch.ones(
                1, 1, *image_chw.shape[2:], dtype=torch.bool
            )  # (B, 1, H, W)

            with fixed_seed(sample_seed):
                random_erase_parameters = T.RandomErasing.get_params(
                    image_chw[0],
                    scale=(0.02, 0.15),
                    ratio=(0.4, 2.5),
                )

            image_chw = F.erase(
                image_chw,
                *random_erase_parameters[:-1],
                0,
            )  # scalar 0 ok
            valid_vector_mask_chw = F.erase(
                valid_vector_mask_chw,
                *random_erase_parameters[:-1],
                0,
            )  # 1 channel
            valid_pixel_mask_chw = F.erase(
                valid_pixel_mask_chw,
                *random_erase_parameters[:-1],
                0,
            )

            image_tensor = image_chw.permute(0, 2, 3, 1).contiguous()  # back to NHWC
            valid_vector = vector[valid_vector_mask_chw[0, 0]]

            # remove rbfovs whose centers fall in erased area
            with ephemeral_cache():
                inside_fov_mask = compute_inside_fov_mask(
                    polar2cartesian(sample_converted_rbfov[:, :2]),
                    valid_vector,
                )
                sample_converted_rbfov = sample_converted_rbfov[inside_fov_mask]

        valid_pixel_mask = valid_pixel_mask_chw[:, 0]  # (1, H, W)

        # Compute the spherical image using the internal method.
        spherical_image = self._equirectangular_projection(image_tensor[0])
        spherical_valid_pixel_mask = self._equirectangular_projection(
            valid_pixel_mask[0].unsqueeze(-1)
        )
        batch_spherical_image = BatchSphericalImage(spherical_image)
        batch_spherical_valid_pixel_mask = BatchSphericalImage(
            spherical_valid_pixel_mask
        )

        output_vector_mask = torch.ones(
            image_tensor.shape[1:3], dtype=torch.bool
        )  # as of now all pixels are valid since input is still panoramic

        output_vector = to_torch(
            generate_equirectangular_normal_map(
                self.downsample_image_size[-1],
                self.downsample_image_size[0],
            )
        )

        panoramic_width, panoramic_height = self.downsample_image_size

        if random_rotation:
            with fixed_seed(sample_seed):
                R = sample_random_rotation_matrices()

            vector = batch_spherical_image.vector
            R = R.squeeze().to(dtype=vector.dtype)

            # rotate batch_spherical_image
            rotated_vector = (R @ vector.T).T

            batch_spherical_image = deepcopy(
                batch_spherical_image
            )  # avoid in-place modification
            original_vector = copy_or_clone(batch_spherical_image.vector)
            batch_spherical_image.vector = rotated_vector
            batch_spherical_valid_pixel_mask.vector = rotated_vector

            # rotate sample_converted_rbfov
            sample_converted_rbfov = self._rotate_converted_rbfovs(
                sample_converted_rbfov, R
            )

            if getattr(self, "output_vector", None) is None:
                # resample into panoramic images
                with ephemeral_cache():
                    # ! not using disable=(seed is None) because accumulated cache will be too large, bigger VRAM may help
                    # turn off cache if seed is None because it's true random rotation and
                    # we are unlikely to encounter the same input vector again
                    batch_spherical_image = self.color_resampler(
                        batch_spherical_image,
                        original_vector,
                    )
                    batch_spherical_valid_pixel_mask = self.label_resampler(
                        batch_spherical_valid_pixel_mask,
                        original_vector,
                    )

                # extract planar images
                image_tensor = batch_spherical_image.batch_value.clone().reshape(
                    1, panoramic_height, panoramic_width, 3
                )
                valid_pixel_mask = batch_spherical_valid_pixel_mask.batch_value.reshape(
                    1, panoramic_height, panoramic_width
                )

        if getattr(self, "output_vector", None) is not None:
            # resample into pinhole/fisheye images
            output_image_height, output_image_width = self.output_vector.shape[:2]
            output_vector_mask = (
                torch.ones((output_image_height, output_image_width), dtype=torch.bool)
                if getattr(self, "output_vector_mask", None) is None
                else self.output_vector_mask
            )

            with ephemeral_cache(ephemeral=random_rotation):
                # if random rotation then will not accumulate cache
                with fixed_seed(sample_seed):
                    (
                        batch_spherical_image,
                        batch_spherical_valid_pixel_mask,
                        sample_converted_rbfov,
                    ) = self._sample_images_from_random_direction(
                        batch_spherical_image,
                        batch_spherical_valid_pixel_mask,
                        sample_converted_rbfov,
                        random_direction=self.random_rotation_probability == 0.0,
                        # if not random rotation then we randomly sample from 1/14 directions
                    )

            # remove out-of-fov rbfovs
            with ephemeral_cache():
                inside_fov_mask = compute_inside_fov_mask(
                    polar2cartesian(sample_converted_rbfov[:, :2]),
                    batch_spherical_image.vector,
                )
                sample_converted_rbfov = sample_converted_rbfov[inside_fov_mask]

            # extract planar images
            image_tensor = torch.zeros(
                (1, output_image_height, output_image_width, 3),
            )
            image_tensor[:, output_vector_mask] = (
                batch_spherical_image.batch_value.clone().reshape(1, -1, 3)
            )
            valid_pixel_mask = torch.zeros(
                (1, output_image_height, output_image_width),
                dtype=torch.bool,
            )
            valid_pixel_mask[:, output_vector_mask] = (
                batch_spherical_valid_pixel_mask.batch_value.clone()
                .to(dtype=torch.bool)
                .reshape(1, -1)
            )

            output_vector = copy_or_clone(self.output_vector)

        # enforce default all true mask for safety
        batch_spherical_image.mask = torch.ones(
            batch_spherical_image.vector.shape[0], dtype=torch.bool
        )
        batch_spherical_valid_pixel_mask.mask = batch_spherical_image.mask

        return {
            "inputs": {
                "images": image_tensor,
                "valid_pixel_masks": valid_pixel_mask,  # shape (1, H, W), different across samples because of potential random erase
                "spherical_images": batch_spherical_image,
                "spherical_valid_pixel_masks": batch_spherical_valid_pixel_mask,  # different across samples because of potential random erase
            },
            "labels": {
                # "raw_rbfovs": [entry["raw_rbfov"]],
                "converted_rbfovs": [sample_converted_rbfov],
            },
            "meta": {
                **self.meta,
                "output_vector": output_vector,  # same across samples
                "output_vector_mask": output_vector_mask,  # same across samples
            },
        }

    @multimethod
    def normalize_rotation_differentiable(self, gamma: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Normalize rotation angles into the range [-pi/2, pi/2] in a differentiable way.
        This would not penalize gamma referring to the same 2D rotation but out of range.

        Args:
            gamma (torch.Tensor): Rotation angles in radians.

        Returns:
            torch.Tensor: Normalized angles in [-pi/2, pi/2].
        """
        angle_sin = torch.sin(gamma)  # Keeps periodicity without hard clipping
        angle_cos = torch.cos(gamma)

        # atan2 maps to [-pi, pi]
        gamma_prime = torch.atan2(angle_sin, angle_cos)

        # Convert gamma_prime to range [-pi/2, pi/2] smoothly
        gamma_prime = gamma_prime - torch.round(gamma_prime / torch.pi) * torch.pi

        return gamma_prime

    @multimethod
    def normalize_rotation_differentiable(self, gamma: np.ndarray) -> np.ndarray:  # type: ignore  # noqa: F811
        """
        Normalize rotation angles into the range [-pi/2, pi/2] in a differentiable way.

        Args:
            gamma (np.ndarray): Rotation angles in radians.

        Returns:
            np.ndarray: Normalized angles in [-pi/2, pi/2].
        """
        gamma = np.asarray(gamma)

        # Compute sin and cos to maintain periodicity
        angle_sin = np.sin(gamma)
        angle_cos = np.cos(gamma)

        # atan2 maps angles to [-pi, pi]
        gamma_prime = np.arctan2(angle_sin, angle_cos)

        # Convert gamma_prime to range [-pi/2, pi/2] smoothly
        gamma_prime = gamma_prime - np.round(gamma_prime / np.pi) * np.pi

        return gamma_prime

    @multimethod
    def normalize_rotation_differentiable(self, gamma: float) -> float:  # noqa: F811
        """
        Normalize rotation angles into the range [-pi/2, pi/2] in a differentiable way.

        Args:
            gamma (float): Rotation angles in radians.

        Returns:
            float: Normalized angles in [-pi/2, pi/2].
        """
        # Compute sin and cos to maintain periodicity
        angle_sin = math.sin(gamma)
        angle_cos = math.cos(gamma)

        # atan2 maps angles to [-pi, pi]
        gamma_prime = math.atan2(angle_sin, angle_cos)

        # Convert gamma_prime to range [-pi/2, pi/2] smoothly
        gamma_prime = gamma_prime - round(gamma_prime / math.pi) * math.pi

        return gamma_prime

    def _rotate_converted_rbfovs(
        self,
        converted_rbfovs: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rotate a single sample's converted_rbfovs using the rotation matrix R.
        Each row is formatted as: [θ, φ, α, β, γ, category].
        1) Rotate the center (θ, φ).
        2) Rotate the in-plane angle γ accordingly.
        """
        if converted_rbfovs.numel() == 0:
            return converted_rbfovs

        _dtype = R.dtype
        _device = R.device

        # Make a safe copy so we don't modify in-place.
        _converted_rbfovs = copy_or_clone(converted_rbfovs).to(
            dtype=_dtype, device=_device
        )

        # --------------------------------------------------------------------------
        # 1) Rotate the center [θ, φ].
        center_vector = polar2cartesian(_converted_rbfovs[:, :2]).to(
            dtype=_dtype, device=_device
        )  # shape (N,3)
        rotated_center_vector = (R @ center_vector.T).T  # (N,3)
        _converted_rbfovs[:, :2] = cartesian2polar(rotated_center_vector).to(
            dtype=_dtype, device=_device
        )

        # --------------------------------------------------------------------------
        # 2) Solve for rotated RBFoV orientation γ
        # R @ R_z @ R_y @ R_x = R_z' @ R_y' @ R_x', solve for R_x'
        # R_x' = R_y'.T @ R_z'.T @ R @ R_z @ R_y @ R_x
        N = _converted_rbfovs.shape[0]

        theta = converted_rbfovs[:, 0]
        phi = converted_rbfovs[:, 1]
        gamma = converted_rbfovs[:, 4]
        theta_prime = _converted_rbfovs[:, 0]
        phi_prime = _converted_rbfovs[:, 1]

        # construct batch matrix one-by-one
        x_angle = gamma
        sin_x, cos_x = torch.sin(x_angle), torch.cos(x_angle)
        R_x = torch.zeros((N, 3, 3), dtype=_dtype, device=_device)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos_x
        R_x[:, 1, 2] = -sin_x
        R_x[:, 2, 1] = sin_x
        R_x[:, 2, 2] = cos_x

        y_angle = -theta
        sin_y, cos_y = torch.sin(y_angle), torch.cos(y_angle)
        R_y = torch.zeros((N, 3, 3), dtype=_dtype, device=_device)
        R_y[:, 0, 0] = cos_y
        R_y[:, 0, 2] = sin_y
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin_y
        R_y[:, 2, 2] = cos_y

        z_angle = phi
        sin_z, cos_z = torch.sin(z_angle), torch.cos(z_angle)
        R_z = torch.zeros((N, 3, 3), dtype=_dtype, device=_device)
        R_z[:, 0, 0] = cos_z
        R_z[:, 0, 1] = -sin_z
        R_z[:, 1, 0] = sin_z
        R_z[:, 1, 1] = cos_z
        R_z[:, 2, 2] = 1

        y_prime_angle = -theta_prime
        sin_y_prime, cos_y_prime = torch.sin(y_prime_angle), torch.cos(y_prime_angle)
        R_y_prime = torch.zeros((N, 3, 3), dtype=_dtype, device=_device)
        R_y_prime[:, 0, 0] = cos_y_prime
        R_y_prime[:, 0, 2] = sin_y_prime
        R_y_prime[:, 1, 1] = 1
        R_y_prime[:, 2, 0] = -sin_y_prime
        R_y_prime[:, 2, 2] = cos_y_prime

        z_prime_angle = phi_prime
        sin_z_prime, cos_z_prime = torch.sin(z_prime_angle), torch.cos(z_prime_angle)
        R_z_prime = torch.zeros((N, 3, 3), dtype=_dtype, device=_device)
        R_z_prime[:, 0, 0] = cos_z_prime
        R_z_prime[:, 0, 1] = -sin_z_prime
        R_z_prime[:, 1, 0] = sin_z_prime
        R_z_prime[:, 1, 1] = cos_z_prime
        R_z_prime[:, 2, 2] = 1

        R_x_prime = R_y_prime.mT @ R_z_prime.mT @ R.unsqueeze(0) @ R_z @ R_y @ R_x

        gamma_prime = torch.atan2(R_x_prime[:, 2, 1], R_x_prime[:, 1, 1])

        # normalize gamma prime to [-pi/2, pi/2]
        gamma_prime = self.normalize_rotation_differentiable(gamma_prime)

        _converted_rbfovs[:, 4] = gamma_prime

        _converted_rbfovs = _converted_rbfovs.to(
            dtype=converted_rbfovs.dtype, device=converted_rbfovs.device
        )

        return _converted_rbfovs

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Custom collate function for PyTorch's DataLoader.

        Args:
            batch (list[dict]): A list of samples, where each sample is a dict as returned by __getitem__.

        Returns:
            dict: A dictionary with keys:
                - "inputs":
                    - "images": a tensor of shape (batch_size, rows, cols, C) created by stacking image tensors.
                    - "spherical_images": a BatchSphericalImage instance built from the list of spherical images.
                - "labels":
                    - "converted_rbfovs": List of list of converted rbfovs
                    # - "raw_rbfovs": List of list of raw rbfovs
                    why double list? each image may have variable number of rbfovs, and I don't wanna do padding
        """
        images = torch.cat([sample["inputs"]["images"] for sample in batch], dim=0)
        valid_pixel_masks = torch.cat(
            [sample["inputs"]["valid_pixel_masks"] for sample in batch], dim=0
        )

        # Build the BatchSphericalImage from the list of per-sample spherical images.
        # Assume that all spherical images share the same geometry (vector and polar).
        batch_spherical_image = BatchSphericalImage(
            batch_value=[sample["inputs"]["spherical_images"] for sample in batch],
        )
        batch_spherical_valid_pixel_mask = BatchSphericalImage(
            batch_value=[
                sample["inputs"]["spherical_valid_pixel_masks"] for sample in batch
            ],
        )

        converted_rbfovs = []
        # raw_rbfovs = []
        for sample in batch:
            label = sample["labels"]
            converted_rbfovs.extend(label["converted_rbfovs"])
            # raw_rbfovs.extend(label["raw_rbfovs"])

        return {
            "inputs": {
                "images": images,
                "valid_pixel_masks": valid_pixel_masks,
                "spherical_images": batch_spherical_image,
                "spherical_valid_pixel_masks": batch_spherical_valid_pixel_mask,
            },
            "labels": {
                "converted_rbfovs": converted_rbfovs,
                # "raw_rbfovs": raw_rbfovs,
            },
            "meta": batch[0]["meta"],  # all meta should be the same
        }

    @staticmethod
    def _rotate_candidate_points(
        vector: torch.Tensor,
        neg_phi: float,
        theta: float,
        neg_gamma: float,
    ) -> torch.Tensor:
        """
        Rotate candidate points (in Cartesian coordinates) by the inverse rotation:
        first about the z-axis by neg_phi, then about the y-axis by neg_theta,
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

        # Rotation about z-axis by -phi.
        cos_z = math.cos(neg_phi)
        sin_z = math.sin(neg_phi)
        R_z = torch.tensor(
            [
                [cos_z, -sin_z, 0],
                [sin_z, cos_z, 0],
                [0, 0, 1],
            ],
            device=_device,
            dtype=_dtype,
        )

        # Rotation about y-axis by -theta.
        cos_y = math.cos(theta)
        sin_y = math.sin(theta)
        R_y = torch.tensor(
            [
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y],
            ],
            device=_device,
            dtype=_dtype,
        )

        # Rotation about x-axis by -gamma.
        cos_x = math.cos(neg_gamma)
        sin_x = math.sin(neg_gamma)
        R_x = torch.tensor(
            [
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x],
            ],
            device=_device,
            dtype=_dtype,
        )

        # Combined rotation: first R_z, then R_y, then R_x.
        R = R_x @ R_y @ R_z
        rotated_vector = (R @ vector.T).T
        return rotated_vector

    def generate_gt_maps(
        self,
        rbfovs_list: list[torch.Tensor],
        vector: torch.Tensor,
        n_points: int = 3,
    ) -> dict[str, BatchSphericalImage]:
        """
        Given ground truth RBFoVs (converted_rbfovs), generate 4 ground truth maps (BatchSphericalImage)
        for supervision: heatmap, center offset, size, angle, mask

        heatmap: classification signal, a BatchSphericalImage with batch_value of shape (B, N, num_categories)
        center offset: offset on θ, φ, a BatchSphericalImage with batch_value of shape (B, N, 2)
        size: angular size on α, β, a BatchSphericalImage with batch_value of shape (B, N, 2)
        angle: orientation on γ, a BatchSphericalImage with batch_value of shape (B, N, 1)
        mask: the mask that dictates regression supervision

        Refer to Fig 5 of https://doi.org/10.1609/aaai.v36i1.19929
                Fig 6 of https://link.springer.com/10.1007/978-3-031-20074-8_14

        Args:
            rbfovs_list (list[torch.Tensor]): the ["labels"]["converted_rbfovs"] from a batch sampled from the dataloader (and moved to device/dtype).
                                                    a list or tensor of shape (num_objects, 6) for each sample.
                                                    Each row has format: [θ, φ, α, β, γ, category]
            vector (torch.Tensor): Tensor of shape (N, 3) containing unit Cartesian
                vectors on the sphere.
            n_points (int, optional): number of nearest points to "lit up" for the regression maps

        Accesses:
            self.category_name_table: list of category names (used to determine num_categories)

        Returns:
            dict: a ground truth map dictionary containing keys: "heatmaps", "center_offsets", "sizes", "rotations", "masks"
        """
        num_categories = len(self.category_name_table)
        N = vector.shape[0]
        _device = vector.device
        polar = cartesian2polar(vector)

        # Initialize ground truth maps as zeros.
        heatmaps, center_offsets, sizes, rotations, masks = [], [], [], [], []

        # Loop over batch samples.
        for rbfovs in rbfovs_list:
            # localize dtype
            _dtype = rbfovs.dtype

            heatmap = torch.zeros((N, num_categories), dtype=_dtype, device=_device)
            center_offset = torch.zeros((N, 2), dtype=_dtype, device=_device)
            size = torch.zeros((N, 2), dtype=_dtype, device=_device)
            angle = torch.zeros((N, 1), dtype=_dtype, device=_device)
            mask = torch.zeros((N, 1), dtype=torch.bool, device=_device)

            # For each ground truth rbfov.
            for rbfov in rbfovs:
                # Each rbfov is [θ, φ, α, β, γ, category] Unpack parameters.
                theta_gt, phi_gt, alpha, beta, gamma, category = rbfov.tolist()[:7]
                category = int(category)

                # === Step 1: Rotate candidate points ===
                # Rotate candidate points by the inverse of the RBFoV’s rotation.
                # That is, apply rotations: first about z by -phi, then y by theta, then x by -gamma.
                rotated_candidates = self._rotate_candidate_points(
                    vector,
                    neg_phi=-phi_gt,
                    theta=theta_gt,
                    neg_gamma=-gamma,
                )

                # === Step 2: Gnomonic projection and BFoV filtering ===
                # In the canonical frame, the tangent plane at (1,0,0) is defined by (y/x, z/x).
                # Only consider points with positive x (i.e. in front of the tangent plane).
                valid_mask = rotated_candidates[:, 0] > 1e-6
                if valid_mask.sum() == 0:
                    continue  # skip if no valid points, practically impossible if in panoramic mode
                valid_indices = torch.nonzero(valid_mask).squeeze()
                valid_candidates = rotated_candidates[valid_mask]
                gnomonic_y = valid_candidates[:, 1] / valid_candidates[:, 0]  # y/x
                gnomonic_z = valid_candidates[:, 2] / valid_candidates[:, 0]  # z/x

                # set "inside" probability such that ~3σ covers the BFoV extent
                allowed_half_width = math.tan(beta / 2)
                allowed_half_height = math.tan(alpha / 2)
                sigma_y = allowed_half_width / 3.0
                sigma_z = allowed_half_height / 3.0

                # Determine inside-mask.
                inside_mask = (gnomonic_y.abs() < allowed_half_width) & (
                    gnomonic_z.abs() < allowed_half_height
                )
                # Force include the candidate closest to (0,0) in the gnomonic plane, helpful when the object is small
                min_idx = torch.argmin(torch.sqrt(gnomonic_y**2 + gnomonic_z**2))
                inside_mask[min_idx] = True

                bfov_indices = valid_indices[inside_mask]

                # Compute the normalized 2D Gaussian probability.
                # ! Note that this is not the true 2D Gaussian probability mathematically, because we want the value to be in [0.0, 1.0], true 2D Gaussian has a denominator
                pseudo_gaussian_probability = torch.exp(
                    -0.5
                    * (
                        (gnomonic_y[inside_mask] / sigma_y) ** 2
                        + (gnomonic_z[inside_mask] / sigma_z) ** 2
                    )
                ).to(heatmap.dtype)
                # Update heatmap for the given category by taking the maximum response.
                heatmap[bfov_indices, category] = torch.max(
                    heatmap[bfov_indices, category], pseudo_gaussian_probability
                )

                # === Regression Targets ===
                # For regression, compute geodesic distances between each polar point and the RBFoV center.
                center_polar = torch.tensor(
                    [theta_gt, phi_gt], dtype=_dtype, device=_device
                )
                distances = spherical_distance(
                    polar, center_polar.repeat((N, 1))
                )  # shape (N,)
                _, closest_indices = torch.topk(
                    distances, k=min(n_points, distances.numel()), largest=False
                )
                # For these points, assign:
                # - Center offset: difference between RBFoV center and polar.
                center_offset[closest_indices] = (
                    center_polar - polar[closest_indices]
                ).to(dtype=_dtype, device=_device)
                # - Size: assign (α, β).
                size[closest_indices] = torch.tensor(
                    [alpha, beta], dtype=_dtype, device=_device
                )
                # - Angle: assign γ.
                angle[closest_indices, 0] = gamma
                # - Update regression mask.
                mask[closest_indices] = True

            heatmaps.append(heatmap)
            center_offsets.append(center_offset)
            sizes.append(size)
            rotations.append(angle)
            masks.append(mask)

        # Construct GT maps using the original vector directly.
        gt_maps = {}
        gt_maps["heatmaps"] = BatchSphericalImage(batch_value=heatmaps, vector=vector)
        gt_maps["center_offsets"] = BatchSphericalImage(
            batch_value=center_offsets, vector=vector
        )
        gt_maps["sizes"] = BatchSphericalImage(batch_value=sizes, vector=vector)
        gt_maps["rotations"] = BatchSphericalImage(batch_value=rotations, vector=vector)
        gt_maps["masks"] = BatchSphericalImage(batch_value=masks, vector=vector)

        return gt_maps

    @staticmethod
    def matrix_nms(
        rbfovs: torch.Tensor,
        sigma: float = 0.5,
        score_threshold: float = 0.5,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies Matrix NMS to unordered RBFoVs for a single category.

        Args:
            rbfovs (torch.Tensor): Tensor of shape (N, 7) where each row is
                [θ, φ, α, β, γ, category, confidence].
            sigma (float, optional): Decay parameter for Gaussian decay (default: 0.5).
            score_threshold (float, optional): Proposals with updated confidence below this are discarded. Defaults to 0.5.

        Returns:
            torch.Tensor: Filtered RBFoVs (shape (K, 7)) sorted in descending order by updated confidence.
        """
        if rbfovs.numel() == 0:
            return rbfovs

        # Sort proposals by confidence (last column index) in descending order.
        scores = rbfovs[:, -1]
        order = torch.argsort(scores, descending=True)
        proposals = rbfovs[order]  # shape: (N, 7)

        # Construct pairwise intersection over union matrix (a confusion matrix with IoU as value)
        iou_matrix = pairwiseIoU(
            proposals,
            proposals,
            *args,
            **kwargs,
        )

        N = proposals.shape[0]
        # Initialize a decay factor vector.
        decay = torch.ones(N, device=proposals.device, dtype=proposals.dtype)

        # For each proposal (starting from index 1), update the decay factor based on
        # the maximum IoU with any higher-scoring proposal.
        for i in range(1, N):
            max_iou = iou_matrix[i, :i].max()
            decay[i] = torch.exp(-(max_iou**2) / sigma)

        # Update scores.
        updated_scores = proposals[:, -1] * decay
        proposals[:, -1] = updated_scores

        # Filter out proposals with low updated confidence.
        keep_mask = updated_scores >= score_threshold
        proposals = proposals[keep_mask]

        # Re-sort proposals by updated confidence in descending order.
        final_order = torch.argsort(proposals[:, -1], descending=True)
        return proposals[final_order]

    @staticmethod
    def matrix_nms_with_backstop(
        rbfovs: torch.Tensor,
        sigma: float = 2.0,
        score_threshold: float = 0.25,
        reduction: str = "min",  # "min" (SOLOv2) or "prod" (stronger in crowds)
        compensate: bool = True,  # SOLOv2-style compensation
        hard_iou_threshold: float = 0.45,  # greedy Hard-NMS backstop IoU threshold
        epsilon_tie_break: float = 1e-6,  # deterministic tie-breaker for IoUs
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies Matrix NMS (Gaussian) to unordered RBFoVs for a single category,
        followed by an optional greedy Hard-NMS backstop to forcibly remove residual overlaps.

        Args:
            rbfovs (torch.Tensor): Tensor of shape (N, 7) where each row is
                [θ, φ, α, β, γ, category, confidence].
            sigma (float, optional):
                Gaussian decay multiplier; larger values produce stronger suppression.
            score_threshold (float, optional):
                Proposals with updated confidence below this are discarded.
            reduction (str, optional):
                Row-wise reduction method across higher-scoring proposals.
                - "min": SOLOv2-style (default).
                - "prod": multiplicative accumulation (stronger suppression).
            compensate (bool, optional):
                Whether to apply max-IoU compensation (SOLOv2-style).
            hard_iou_threshold (float, optional):
                After Matrix-NMS, run greedy Hard-NMS with this IoU cutoff.
                Set to 1.0 to effectively disable the backstop.
            epsilon_tie_break (float, optional):
                Small deterministic perturbation added to IoU values
                to break exact ties and prevent compensation cancellation. Defaults to 1e-6.

        Returns:
            torch.Tensor: Filtered RBFoVs (shape (K, 7)) sorted in descending order
            by updated confidence after suppression.
        """
        if rbfovs.numel() == 0:
            return rbfovs
        if sigma <= 0:
            raise ValueError(f"`sigma` must be positive, got {sigma}.")
        if reduction not in ("min", "prod"):
            raise ValueError(f"`reduction` must be 'min' or 'prod', got {reduction}.")
        if not (0.0 < hard_iou_threshold <= 1.0):
            raise ValueError(
                f"`hard_iou_threshold` must be in (0,1], got {hard_iou_threshold}."
            )

        # ---------------------------------------------------------------------
        # Step 1: Sort proposals by confidence (descending)
        # ---------------------------------------------------------------------
        confidence_scores = rbfovs[:, -1]
        sort_order_descending = torch.argsort(confidence_scores, descending=True)
        sorted_proposals = rbfovs[sort_order_descending]

        # ---------------------------------------------------------------------
        # Step 2: Compute pairwise IoU between sorted proposals
        # ---------------------------------------------------------------------
        pairwise_iou_matrix = pairwiseIoU(
            sorted_proposals, sorted_proposals, *args, **kwargs
        )
        num_proposals = sorted_proposals.shape[0]
        device, dtype = sorted_proposals.device, sorted_proposals.dtype

        # deterministic tie-break (prevents identical IoUs from canceling suppression)
        if epsilon_tie_break > 0.0:
            rank = torch.arange(num_proposals, device=device, dtype=dtype)
            pairwise_iou_matrix = pairwise_iou_matrix + epsilon_tie_break * (
                rank.view(-1, 1) - rank.view(1, -1)
            )

        # ---------------------------------------------------------------------
        # Step 3: Identify higher-scoring neighbors for each row
        # ---------------------------------------------------------------------
        higher_mask = torch.tril(
            torch.ones((num_proposals, num_proposals), device=device, dtype=torch.bool),
            diagonal=-1,
        )

        # ---------------------------------------------------------------------
        # Step 4: Compute Gaussian decays
        # ---------------------------------------------------------------------
        gaussian_decay_matrix = torch.exp(-sigma * (pairwise_iou_matrix**2)).to(
            dtype=dtype
        )

        if compensate:
            iou_higher_only = torch.where(
                higher_mask,
                pairwise_iou_matrix,
                torch.zeros_like(pairwise_iou_matrix, device=device, dtype=dtype),
            )
            max_iou_to_higher, _ = iou_higher_only.max(dim=1)
            compensate_decay = torch.exp(-sigma * (max_iou_to_higher**2)).clamp_min(
                1e-8
            )
            gaussian_decay_matrix = gaussian_decay_matrix / compensate_decay.view(-1, 1)

        # ---------------------------------------------------------------------
        # Step 5: Reduction (min or prod) across higher neighbors
        # ---------------------------------------------------------------------
        if reduction == "min":
            masked_decay = torch.where(
                higher_mask,
                gaussian_decay_matrix,
                torch.ones_like(gaussian_decay_matrix, dtype=dtype, device=device),
            )
            rowwise_decay = masked_decay.min(dim=1).values
        else:  # "prod"
            masked_log = torch.where(
                higher_mask,
                torch.log(gaussian_decay_matrix.clamp_min(1e-8)),
                torch.zeros_like(gaussian_decay_matrix, dtype=dtype, device=device),
            )
            rowwise_decay = torch.exp(masked_log.sum(dim=1)).clamp(0.0, 1.0)

        has_higher = higher_mask.any(dim=1)
        rowwise_decay = torch.where(
            has_higher,
            rowwise_decay,
            torch.ones_like(rowwise_decay, dtype=dtype, device=device),
        )
        rowwise_decay[0] = torch.tensor(1.0, dtype=dtype, device=device)

        # ---------------------------------------------------------------------
        # Step 6: Apply decay and filter by confidence
        # ---------------------------------------------------------------------
        updated_confidence_scores = sorted_proposals[:, -1] * rowwise_decay
        keep_mask = updated_confidence_scores >= score_threshold
        if not torch.any(keep_mask):
            return sorted_proposals.new_empty((0, sorted_proposals.shape[1]))

        filtered = sorted_proposals[keep_mask].clone()
        filtered[:, -1] = updated_confidence_scores[keep_mask]

        # ---------------------------------------------------------------------
        # Step 7: Greedy Hard-NMS backstop (optional)
        # ---------------------------------------------------------------------
        if hard_iou_threshold < 1.0 and filtered.size(0) > 1:
            sorted_idx = torch.argsort(filtered[:, -1], descending=True)
            filtered = filtered[sorted_idx]
            iou_submatrix = pairwiseIoU(filtered, filtered, *args, **kwargs)

            keep = []
            suppressed = torch.zeros(filtered.size(0), dtype=torch.bool, device=device)
            for i in range(filtered.size(0)):
                if suppressed[i]:
                    continue
                keep.append(i)
                suppressed |= iou_submatrix[i] >= hard_iou_threshold

            filtered = filtered[torch.tensor(keep, device=device, dtype=torch.long)]

        # ---------------------------------------------------------------------
        # Step 8: Return final proposals sorted by updated confidence
        # ---------------------------------------------------------------------
        final_order = torch.argsort(filtered[:, -1], descending=True)
        return filtered[final_order]

    @staticmethod
    def non_maximum_suppression(
        rbfovs: list[torch.Tensor],
        max_rbfov_per_category: int = 20,
        *args,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Non maximum suppression.

        Args:
            rbfovs (list[torch.Tensor]): list of variable number of RBFoVs, len(rbfovs) = batch_size
                                            each tensor's shape: (num_rbfovs, 6 or 7)
                                            each row's format: [θ, φ, α, β, γ, category(, probability)]
            max_rbfov_per_category (int, optional): Maximum number of detections to retain per category.
                                                    Defaults to 20.

        Returns:
            list[torch.Tensor]: RBFoVs with non-maximum detection suppressed
        """
        suppressed_rbfovs = []
        for sample in rbfovs:
            # Initialize filtered_detections as an empty tensor with shape (0, 7) on the same device/dtype as sample.
            filtered_detections = sample.new_empty((0, 7))

            # If sample is empty, unique_categories will be an empty tensor.
            unique_categories = (
                sample[:, 5].unique() if sample.numel() > 0 else sample.new_empty((0,))
            )
            for category in unique_categories:
                # Select all detections for this category.
                category_mask = sample[:, 5] == category
                detections_for_category = sample[category_mask]

                # filtered_detections = torch.vstack(
                #     (
                #         filtered_detections,
                #         PandoraDataset.matrix_nms(
                #             detections_for_category, *args, **kwargs
                #         )[:max_rbfov_per_category],
                #     )
                # )
                filtered_detections = torch.vstack(
                    (
                        filtered_detections,
                        PandoraDataset.matrix_nms_with_backstop(
                            detections_for_category, *args, **kwargs
                        )[:max_rbfov_per_category],
                    )
                )

            suppressed_rbfovs.append(filtered_detections)
        return suppressed_rbfovs

    def extract_raw_rbfovs(
        self,
        maps: dict[str, BatchSphericalImage],
        probability_threshold: float = 0.01,
        pool_radius: float = None,
    ) -> list[torch.Tensor]:
        """
        Given 4 maps from GT/model, extract (variable number of) raw converted_rbfovs for downstream non-maximum-suppression (NMS).
        Ideally, we want a 1-to-1 mapping between maps and RBFoVs (including NMS).

        Args:
            maps (dict[str, BatchSphericalImage]): A map dictionary with keys produced by generate_gt_maps or the object RBFoV model.
                Expected keys are:
                    "heatmaps": BatchSphericalImage with batch_value of shape (B, N, num_categories)
                    "center_offsets": BatchSphericalImage with batch_value of shape (B, N, 2)
                    "sizes": BatchSphericalImage with batch_value of shape (B, N, 2)
                    "rotations": BatchSphericalImage with batch_value of shape (B, N, 1)
                All share the same geometry (vector and polar).
            probability_threshold (float, optional): Minimum probability threshold in [0.0, 1.0) for a peak to be considered valid. Defaults to 0.1.
            pool_radius (float, optional): If provided, uses this fixed pooling radius (in radians). If None, it is computed as 1.5 x average nearest neighbor angular distance.

        Returns:
            list[torch.Tensor]: A list (length B) of tensors (each shape (num_rbfovs, 7)) containing: [θ, φ, α, β, γ, category, probability].
        """
        assert (
            0.0 <= probability_threshold < 1.0
        ), f"Expected probability_threshold in [0.0, 1.0), got {probability_threshold}"
        polar = maps["heatmaps"].polar
        _device, _dtype = (
            maps["heatmaps"].batch_value.device,
            maps["heatmaps"].batch_value.dtype,
        )

        # If no fixed pool_radius is provided, compute it automatically.
        if pool_radius is None:
            pool_radius = (
                1.5 * nearest_neighbor_distance(polar).mean().item()
            )  # ! 1.5 is from heuristics

        circle_pool = CirclePool(
            pool_type="max",
            radius=pool_radius,
            identical_output_vector=True,
            device=_device,
            dtype=_dtype,
        )  # don't worry, the cache is in class-level, pretty fast here

        # shape: (B, N, num_categories)
        original_heatmaps = maps["heatmaps"].batch_value
        pooled_heatmaps = circle_pool(maps["heatmaps"]).batch_value
        local_max_masks = torch.isclose(original_heatmaps, pooled_heatmaps, atol=1e-6)

        raw_batch_rbfovs = []
        for heatmap, center_offset, size, angle, local_max_mask in zip(
            maps["heatmaps"].batch_value,
            maps["center_offsets"].batch_value,
            maps["sizes"].batch_value,
            maps["rotations"].batch_value,
            local_max_masks,
        ):

            all_category_rbfovs = torch.empty((0, 7), device=_device, dtype=_dtype)
            # Process each category channel
            for category_id in range(len(self.category_name_table)):
                category_probability_map = heatmap[:, category_id]  # (N,)
                # Combine threshold and local maximum condition.
                valid_mask = (
                    (category_probability_map >= probability_threshold)
                    & local_max_mask[:, category_id]
                    & ((size[:, 0] > 0) & (size[:, 1] > 0))
                )
                if valid_mask.sum() > 0:
                    # Predicted center: polar + center_offset
                    extract_center = polar[valid_mask, :] + center_offset[valid_mask, :]
                    extract_center = normalize_polar(extract_center)
                    extract_size = torch.clamp(size[valid_mask, :], 0, 2 * torch.pi)
                    extract_angle = self.normalize_rotation_differentiable(
                        angle[valid_mask, :]
                    )
                    extract_category = torch.full(
                        (valid_mask.sum(), 1),
                        float(category_id),
                        device=_device,
                        dtype=_dtype,
                    )
                    extract_probability = (
                        category_probability_map[valid_mask]
                        .unsqueeze(1)
                        .clamp(0.0, 1.0)
                    )
                    # Concatenate to form a tensor of shape (K, 7): [θ, φ, α, β, γ, category, probability]
                    rbfovs = torch.hstack(
                        [
                            extract_center,
                            extract_size,
                            extract_angle,
                            extract_category,
                            extract_probability,
                        ]
                    )
                    # Sort by probability descending
                    rbfovs = rbfovs[torch.argsort(rbfovs[:, -1], descending=True)]
                    all_category_rbfovs = torch.vstack((all_category_rbfovs, rbfovs))
            raw_batch_rbfovs.append(all_category_rbfovs)

        return raw_batch_rbfovs

    @staticmethod
    def move_batch_to(batch: dict, *args, **kwargs) -> dict:
        """
        Moves all tensors in the batch to the specified device and dtype.

        Args:
            batch (dict): The batch dictionary returned by `collate_fn`.

        Returns:
            dict: A batch where all tensors are moved to the specified device and dtype.
        """

        def move_object(x):
            """Helper function to move a tensor/spherical object to the specified device and dtype."""
            if isinstance(x, (torch.Tensor, SphericalImage, BatchSphericalImage)):
                return x.to(*args, **kwargs)
            return x

        def move_nested_structure(data):
            """Recursively move tensors in nested dictionaries or lists."""
            if isinstance(data, dict):
                return {
                    key: move_nested_structure(value) for key, value in data.items()
                }
            elif isinstance(data, list):
                return [move_nested_structure(item) for item in data]
            return move_object(data)

        return move_nested_structure(batch)

    @ephemeral_cache()
    def visualize_batch(
        self,
        batch: dict,
        num_vis: int = -1,
        point_spacing: float = 2 * np.pi / 2000,
        stroke: int = 4,
        remove_oo_fov_rbfovs: bool = False,
    ) -> tuple[list[plt.Figure], list[SphericalImage]]:
        """visualize a batch from the dataset or model output

        Args:
            batch (dict): has keys:
                - "inputs":
                    - "images": a tensor of shape (batch_size, rows, cols, C) created by stacking image tensors.
                    - "spherical_images": a BatchSphericalImage instance built from the list of spherical images.
                - "labels":
                    - "converted_rbfovs": List of list of converted rbfovs
                    # - "raw_rbfovs": List of list of raw rbfovs
                    why double list? each image may have variable number of rbfovs, and I don't wanna do padding
            num_vis (int, optional): number of samples to visualize. Defaults to -1.
            point_spacing (float, optional): geodesic distance between neighbor points on a line of a RBFoV. Defaults to 2 pi / 5000
            stroke (int, optional): width of pixels to mark when mapping the RBFoV on the sphere back to planar image. Defaults to 4.
            remove_oo_fov_rbfovs (bool, optional): whether to remove out-of-fov rbfovs in the spherical visualization. Defaults to False.

        Returns:
            list[plt.Figure]: figure with marked RBFoV with legends, categories are differentiated with colors
            list[SphericalImage]: spherical images with marked RBFoV, cannot use BatchSphericalImage because RBFoV vecs are different
        """
        num_vis = (
            np.clip(num_vis, 1, len(batch["inputs"]["spherical_images"]))
            if num_vis != -1
            else num_vis
        )

        # pure virtual function style that has no side effects
        _images = to_numpy(
            copy_or_clone(
                batch["inputs"]["images"]
                if num_vis == -1
                else batch["inputs"]["images"][:num_vis, ...]
            )
        )
        _converted_rbfovs = [
            to_numpy(copy_or_clone(converted_rbfov))
            for converted_rbfov in (
                batch["labels"]["converted_rbfovs"]
                if num_vis == -1
                else batch["labels"]["converted_rbfovs"][:num_vis]
            )
        ]
        _batch_spherical_image = (
            batch["inputs"]["spherical_images"][:num_vis].to_numpy().copy()
            if num_vis != -1
            else batch["inputs"]["spherical_images"].to_numpy().copy()
        )
        _mask = to_numpy(copy_or_clone(batch["meta"]["output_vector_mask"])).astype(
            bool
        )

        figs: list[plt.Figure] = []
        spherical_images_vis: list[SphericalImage] = []
        for image, _converted_rbfov, spherical_image in list(
            zip(_images, _converted_rbfovs, _batch_spherical_image)
        ):
            spherical_image_vis = spherical_image.clone()

            # flatten for easy colorization
            H, W, C = image.shape
            image_flat = copy_or_clone(spherical_image.value)

            # Track categories present in this image for legend.
            categories_in_image = set()

            # variable number of points in each rbfov so not easy to vmap unless padded
            for rbfov in _converted_rbfov:
                # bbox format: (θ, φ, α, β, γ, category_id): (center | size | rotation | category)
                theta, phi, alpha, beta, gamma, category_id = rbfov[:6]
                category_id = int(category_id)
                categories_in_image.add(category_id)
                # RGBA
                bbox_color = np.array(self.category_colormap[category_id])

                # 1) construct rbfov points in cartesian, centered at (1, 0, 0)
                num_vertical_points = max(2, int(np.ceil(alpha / point_spacing)))
                num_horizontal_points = max(2, int(np.ceil(beta / point_spacing)))

                # construct lines
                vertical_line = np.linspace(
                    -alpha / 2,
                    alpha / 2,
                    num_vertical_points,
                    endpoint=True,
                    dtype=np.float32,
                ).reshape(-1, 1)
                horizontal_line = np.linspace(
                    -beta / 2,
                    beta / 2,
                    num_horizontal_points,
                    endpoint=True,
                    dtype=np.float32,
                ).reshape(-1, 1)

                # Construct left & right edges.
                left_side = np.hstack(
                    (vertical_line, np.full((vertical_line.shape[0], 1), -beta / 2))
                )
                right_side = np.hstack(
                    (vertical_line, np.full((vertical_line.shape[0], 1), beta / 2))
                )
                # Construct top & bottom edges.
                top_side = np.hstack(
                    (
                        np.full((horizontal_line.shape[0], 1), alpha / 2),
                        horizontal_line,
                    )
                )
                bottom_side = np.hstack(
                    (
                        np.full((horizontal_line.shape[0], 1), -alpha / 2),
                        horizontal_line,
                    )
                )

                # assemble & remove replicates
                rbfov_polar = np.vstack((left_side, right_side, top_side, bottom_side))
                rbfov_polar = np.unique(rbfov_polar, axis=0)

                # cast to cartesian
                rbfov_vector = polar2cartesian(rbfov_polar)

                # rotate wrt γ
                row = gamma
                R_x_row = np.array(
                    [
                        [1, 0, 0],
                        [0, math.cos(row), -math.sin(row)],
                        [0, math.sin(row), math.cos(row)],
                    ]
                )
                rbfov_vector = (R_x_row @ rbfov_vector.T).T

                # 2) rotate to its center using axis angle

                # rotate along y axis
                pitch = -theta  # right hand system, positive rotation is negative theta
                R_y_pitch = np.array(
                    [
                        [math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)],
                    ]
                )

                # rotate along z axis
                yaw = phi
                R_z_yaw = np.array(
                    [
                        [math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1],
                    ]
                )

                # rotate!
                rbfov_vector = (R_z_yaw @ R_y_pitch @ rbfov_vector.T).T

                # remove out-of-fov rbfov line points
                if remove_oo_fov_rbfovs:
                    inside_mask = compute_inside_fov_mask(
                        rbfov_vector, spherical_image.vector
                    )
                    rbfov_vector = rbfov_vector[inside_mask]

                # neighbor_indices shape: (rbfov_vector.shape[0], stroke)
                neighbor_indices, _ = nearest_point(
                    rbfov_vector,
                    spherical_image.vector,
                    stroke,
                )

                # flatten and remove duplicates
                bbox_pixel_indices = np.unique(neighbor_indices.flatten())

                # alpha blending bounding box in the planar image
                # ! assumes 255 is the max intensity level
                image_flat[bbox_pixel_indices] = (
                    bbox_color[-1] * bbox_color[:3] * 255
                    + (1 - bbox_color[-1]) * image_flat[bbox_pixel_indices]
                ).astype(image_flat.dtype)

                spherical_image_vis += SphericalImage(
                    value=np.tile(bbox_color[:3] * 255, (rbfov_vector.shape[0], 1)),
                    vector=rbfov_vector,
                )

            vis_image = np.zeros((H, W, C), dtype=image_flat.dtype)
            vis_image[_mask] = image_flat

            # Create a matplotlib figure.
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(vis_image.astype(np.uint8))
            ax.axis("off")

            # Create legend patches for the categories present.
            legend_handles = []
            for cat_id in sorted(categories_in_image):
                label_name = self.category_name_table[cat_id]
                color = self.category_colormap[cat_id]
                patch = patches.Patch(color=color, label=label_name)
                legend_handles.append(patch)

            if legend_handles:
                # Choose number of columns (e.g., 5 per row)
                ncol = min(len(legend_handles), 5)
                # Place the legend with bbox_to_anchor to push it further down.
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.1),
                    ncol=ncol,
                    fontsize="small",
                    frameon=False,
                )

            figs.append(fig)
            spherical_images_vis.append(spherical_image_vis)

        return figs, spherical_images_vis


class PandoraDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the PANDORA object detection dataset.

    Wraps ``PandoraDataset`` for training and validation with distributed
    sampling support. The ``@`` operator (``n @ datamodule``) sub-samples
    both splits.
    """

    def __init__(
        self,
        dataset_base_path: str,
        downsample_image_size: tuple,
        batch_size: int,
        num_workers: int,
        train_augmentation: dict[str, float] | None = None,
        val_augmentation: dict[str, float] | None = None,
        train_output_vector: np.ndarray | None = None,
        train_output_vector_mask: np.ndarray | None = None,
        val_output_vector: np.ndarray | None = None,
        val_output_vector_mask: np.ndarray | None = None,
        val_seed: str | None = None,
        meta: dict | None = None,
    ) -> None:
        """Initialize the PANDORA data module.

        Args:
            dataset_base_path (str): Root directory of the PANDORA dataset.
            downsample_image_size (tuple): Target (width, height) for image downsampling.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of DataLoader worker processes.
            train_augmentation (dict[str, float] | None, optional): Augmentation config for training. Defaults to None.
            val_augmentation (dict[str, float] | None, optional): Augmentation config for validation. Defaults to None.
            train_output_vector (np.ndarray | None, optional): Output grid for training reprojection. Defaults to None.
            train_output_vector_mask (np.ndarray | None, optional): Mask for training output grid. Defaults to None.
            val_output_vector (np.ndarray | None, optional): Output grid for validation reprojection. Defaults to None.
            val_output_vector_mask (np.ndarray | None, optional): Mask for validation output grid. Defaults to None.
            val_seed (str | None, optional): Deterministic seed for validation augmentation. Defaults to None.
            meta (dict | None, optional): Meta dictionary (mean/std, class weights). Defaults to None.
        """
        super().__init__()
        self.dataset_base_path = dataset_base_path

        self.train_augmentation = train_augmentation or {}
        self.val_augmentation = val_augmentation or {}
        self.train_output_vector = train_output_vector
        self.train_output_vector_mask = train_output_vector_mask
        self.val_output_vector = val_output_vector
        self.val_output_vector_mask = val_output_vector_mask
        self.val_seed = val_seed
        self.meta = meta or {}
        self.downsample_image_size = downsample_image_size

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """No-op: PANDORA data files are expected to already be on disk."""

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train and validation datasets if not already set.

        Args:
            stage (str | None, optional): Lightning stage string. Unused. Defaults to None.
        """
        if (
            getattr(self, "train_dataset", None) is None
        ):  # Avoid re-initialization if already set by x @ PandoraDataModule
            self.train_dataset = PandoraDataset(
                dataset_base_path=self.dataset_base_path,
                dataset_type="train",
                augmentation=self.train_augmentation,
                output_vector=self.train_output_vector,
                output_vector_mask=self.train_output_vector_mask,
                downsample_image_size=self.downsample_image_size,
                meta=self.meta,
            )
        if (
            getattr(self, "val_dataset", None) is None
        ):  # Avoid re-initialization if already set by x @ PandoraDataModule
            self.val_dataset = PandoraDataset(
                dataset_base_path=self.dataset_base_path,
                dataset_type="test",
                augmentation=self.val_augmentation,
                output_vector=self.val_output_vector,
                output_vector_mask=self.val_output_vector_mask,
                downsample_image_size=self.downsample_image_size,
                meta=self.meta,
                seed=self.val_seed,
            )

    def __rmatmul__(self, other: int | float) -> PandoraDataModule:
        """Supports x @ PandoraDataModule, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Returns:
            PandoraDataModule
        """
        datamodule_copy = deepcopy(self)
        datamodule_copy.train_dataset = other @ self.train_dataset
        datamodule_copy.val_dataset = other @ self.val_dataset

        return datamodule_copy

    def train_dataloader(self) -> DataLoader:
        """Build the training DataLoader, using DistributedSampler when in DDP/FSDP.

        Returns:
            DataLoader: Shuffled, drop-last DataLoader over the training split.
        """
        if distributed.is_available() and distributed.is_initialized():
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                sampler=DistributedSampler(
                    self.train_dataset,
                    shuffle=True,
                    drop_last=True,
                ),
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                multiprocessing_context="spawn",  # For FAISS GPU in dataloader workers
                collate_fn=self.train_dataset.collate_fn,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                multiprocessing_context="spawn",  # For FAISS GPU in dataloader workers
                collate_fn=self.train_dataset.collate_fn,
            )

    def val_dataloader(self) -> DataLoader:
        """Build the validation DataLoader, using DistributedSampler when in DDP/FSDP.

        Returns:
            DataLoader: Non-shuffled, drop-last DataLoader over the validation split.
        """
        if distributed.is_available() and distributed.is_initialized():
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                sampler=DistributedSampler(
                    self.val_dataset,
                    shuffle=False,
                    drop_last=True,
                ),
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                multiprocessing_context="spawn",  # For FAISS GPU in dataloader workers
                collate_fn=self.val_dataset.collate_fn,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=True,
                multiprocessing_context="spawn",  # For FAISS GPU in dataloader workers
                collate_fn=self.val_dataset.collate_fn,
            )
