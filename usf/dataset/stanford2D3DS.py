#!/usr/bin/env python3
#
# Created on Wed Apr 02 2025 17:37:08
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import os.path as osp
import warnings
from copy import deepcopy
from glob import glob
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import patches
from PIL import Image
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import functional as F

from usf.generate_lens_normal_map import generate_equirectangular_normal_map
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.cache import ephemeral_cache
from usf.utils.files import parse_path, read_file
from usf.utils.spherical import (
    azim_elev_to_rotation_matrix,
    sample_random_rotation_matrices,
)
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import (
    copy_or_clone,
    fixed_seed,
    string_to_seed,
    to_numpy,
    to_torch,
)


class Stanford2D3DSDataset(Dataset):
    """Stanford 2D-3D-S spherical semantic segmentation dataset.

    Each sample is a 360° equirectangular panorama paired with per-pixel semantic
    labels. Images are loaded, optionally downsampled, projected onto the sphere,
    and augmented with random rotation, color jitter, and horizontal flip. An
    optional ``output_vector`` allows resampling onto a different spherical grid.

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
        Initializes the Stanford 2D-3D-S dataset

        Args:
            dataset_base_path (str): Base directory where the Stanford 2D-3D-S dataset is stored
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
                    T.GaussianBlur(kernel_size=(15, 15), sigma=(1.5, 2.5)),
                ],
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

        if dataset_type == "train":
            selected_areas = [1, 2, 3, 4, 6]
        elif dataset_type == "test":
            selected_areas = [5]
        else:
            raise ValueError(
                f"dataset_type must be 'train' or 'test', got {dataset_type}"
            )

        self.image_paths: list[str] = []
        for area in selected_areas:
            self.image_paths.extend(
                glob(
                    osp.join(dataset_base_path, f"area_{area}*", "pano", "rgb", "*.png")
                )
            )
        self.image_paths.sort()

        expected_label_path = osp.join(dataset_base_path, "semantic_labels.json")
        label_path = parse_path(expected_label_path)
        assert label_path, f"label doesn't exist at {expected_label_path}"

        labels = read_file(label_path)

        unique_class_names = sorted(
            set(self.extract_class_name(label) for label in labels)
        )
        self.class_id_to_name = {i: name for i, name in enumerate(unique_class_names)}
        class_name_to_id = {name: i for i, name in enumerate(unique_class_names)}
        label_to_class_id = {
            i: class_name_to_id[self.extract_class_name(label)]
            for i, label in enumerate(labels)
        }
        self.lut = np.zeros(
            (1 << 24,), dtype=np.uint8
        )  # 0 is the safe default <UNK> class
        for label_index, class_id in label_to_class_id.items():
            self.lut[label_index] = class_id

        self.meta = meta or {}
        self.seed = seed

    def set_output_vector(
        self, output_vector: torch.Tensor | np.ndarray | None
    ) -> Stanford2D3DSDataset:
        """Set or clear a fixed output grid for reprojection.

        Args:
            output_vector (torch.Tensor | np.ndarray | None): Unit vectors of
                shape (N, 3), or None to clear.

        Returns:
            Stanford2D3DSDataset: ``self`` for chaining.
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
    ) -> Stanford2D3DSDataset:
        """Set or clear a validity mask for the output grid.

        Args:
            output_vector_mask (torch.Tensor | np.ndarray | None): Boolean or
                float mask of shape (N,), or None to clear.

        Returns:
            Stanford2D3DSDataset: ``self`` for chaining.
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

    @torch.inference_mode()
    def compute_rgb_mean_std(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-channel mean and standard deviation over the dataset (no augmentation),
        using only valid pixels (non-black borders) from RGB images.

        Args:
            stride (int, optional): process every `stride`-th image to speed up (1 = all images). Defaults to 1.

        Returns:
            tuple[np.ndarray, np.ndarray]: (mean[3], std[3]) on [0,1] scale as np.float64 arrays.
        """
        sum_channel = np.zeros(3, dtype=np.float64)
        squared_sum_channel = np.zeros(3, dtype=np.float64)
        total_pixel_count = 0

        for image_index, rgb_path in enumerate(self.image_paths):
            if (image_index % stride) != 0:
                continue

            rgb_image = read_file(rgb_path)  # expected (H, W, 4) uint8
            if rgb_image.dtype != np.uint8:  # ensure uint8
                rgb_image = rgb_image.astype(np.uint8, copy=False)
            if rgb_image.shape[-1] >= 3:  # drop alpha if exists
                rgb_image = rgb_image[..., :3]
            else:
                raise ValueError(
                    f"Expected at least 3 channels in {rgb_path}, got {rgb_image.shape}"
                )

            # Mask out black borders
            valid_pixel_mask = (rgb_image != 0).any(axis=-1)  # (H, W) bool
            if not valid_pixel_mask.any():
                continue

            # Select valid pixels and cast to float64 in [0,1]
            valid_pixels = (
                rgb_image[valid_pixel_mask].astype(np.float64) / 255.0
            )  # (P,3)

            # Accumulate sums
            sum_channel += valid_pixels.sum(axis=0)
            squared_sum_channel += (valid_pixels * valid_pixels).sum(axis=0)
            total_pixel_count += valid_pixels.shape[0]

        if total_pixel_count == 0:
            # Degenerate case: no valid pixels found
            return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)

        mean_channel = sum_channel / total_pixel_count
        variance_channel = (
            squared_sum_channel / total_pixel_count - mean_channel * mean_channel
        )
        std_channel = np.sqrt(np.maximum(variance_channel, 1e-12))
        return mean_channel, std_channel

    @torch.inference_mode()
    def _compute_class_histogram(
        self, ignore_index: int | None = 0, stride: int = 1
    ) -> np.ndarray:
        """
        Count per-class pixels over the dataset (no augmentation).

        Args:
            ignore_index (int | None, optional): if not None, zero out that class count (e.g., 0='<UNK>'). Defaults to 0.
            stride (int, optional): process every `stride`-th image to speed up (1 = all images). Defaults to 1.

        Returns:
            np.ndarray: np.ndarray of counts in shape (self.num_classes,)
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)

        for image_index, rgb_path in enumerate(self.image_paths):
            if (image_index % stride) != 0:
                continue

            semantic_path = rgb_path.replace("/rgb/", "/semantic/", 1).replace(
                "rgb.png", "semantic.png", 1
            )
            semantic = read_file(semantic_path)  # (H, W, 4) uint8

            # mask out black borders
            rgb = read_file(rgb_path)
            black_border_mask = (rgb[..., :3] != 0).any(axis=-1)
            semantic[~black_border_mask, :3] = 0  # set to <UNK>

            class_map = self.semantic_to_class_map(semantic)  # (H, W)
            bincount = np.bincount(
                class_map.reshape(-1), minlength=self.num_classes
            ).astype(np.int64)
            counts += bincount

        if ignore_index is not None:
            counts[int(ignore_index)] = 0

        return counts

    @torch.inference_mode()
    def compute_class_weights(
        self,
        method: Literal[
            "median_frequency",
            "inverse_frequency",
            "effective_number",
        ] = "median_frequency",
        beta: float = 1 - 1e-8,
        clip: tuple[float, float] | None = None,
        normalize_to_num_valid_classes: bool = True,
        ignore_index: int | None = 0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Build a (self.num_classes,) class weight tensor from training set statistics.

        Args:
            method (Literal["median_frequency", "inverse_frequency", "effective_number"], optional): weighting scheme. Defaults to "median_frequency".
            beta (float, optional): parameter for effective_number method. Defaults to 1 - 1e-8.
            clip (tuple[float, float] | None, optional): (min,max) clip for stability, e.g. (0.25, 4.0). Defaults to None.
            normalize_to_num_valid_classes (bool, optional): rescale so mean weight ~ 1. Defaults to True.
            ignore_index (int | None, optional): class ID to ignore (set weight=0). Defaults to 0.
            eps (float, optional): should always be smaller than the residual in beta, i.e. eps < 1 - beta. Defaults to 1e-10.

        Returns:
            torch.Tensor: class weights in shape (self.num_classes,)
        """
        counts = self._compute_class_histogram(
            ignore_index=ignore_index,
        ).astype(np.float64)

        if method == "median_frequency":
            nonzero = counts[counts > 0]
            median = np.median(nonzero) if nonzero.size > 0 else 1.0
            weights = median / np.maximum(counts, eps)

        elif method == "inverse_frequency":
            weights = 1.0 / np.maximum(counts, eps)

        elif method == "effective_number":
            weights = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))

        else:
            raise ValueError(f"Unknown method: {method}")

        if clip is not None:
            low, high = clip
            weights = np.clip(weights, low, high)

        if ignore_index is not None:
            weights[int(ignore_index)] = 0.0

        if normalize_to_num_valid_classes:
            valid = weights > 0
            sum = np.sum(weights[valid]) + eps
            weights *= valid.sum() / sum

        return torch.tensor(weights)

    @property
    def class_weight(self) -> torch.Tensor:
        """Lazily-computed inverse-frequency class weights for loss balancing.

        Returns:
            torch.Tensor: Weight per class, shape (num_classes,).
        """
        if getattr(self, "_class_weight", None) is None:
            self._class_weight = self.compute_class_weights()
        return self._class_weight

    def __rmatmul__(self, other: int | float) -> Stanford2D3DSDataset:
        """Supports x @ Stanford2D3DSDataset, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Raises:
            TypeError: If `other` is not an int or float.

        Returns:
            Stanford2D3DSDataset
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
        dataset_copy.image_paths = [self.image_paths[i] for i in random_indices]

        return dataset_copy

    @staticmethod
    def _get_category_colormap(num_categories: int, alpha=0.8) -> dict:
        cmap = plt.get_cmap("hsv", num_categories)
        colors = {i: cmap(i, alpha) for i in range(num_categories)}
        return colors

    @staticmethod
    def extract_class_name(label: str) -> str:
        """Extract class name from raw label

        Args:
            label (str): raw label

        Returns:
            str: class name
        """
        return label.split("_")[0]

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

    def semantic_to_class_map(self, semantic_image: np.ndarray) -> np.ndarray:
        """
        Converts a semantic RGB image (H, W, 3, dtype=uint8) into a single-channel label map using a precomputed LUT.

        Args:
            semantic_image (np.ndarray): The semantic image (H, W, 3) with RGB encoding.

        Accesses:
            lut (np.ndarray): look up table to convert 24-bit pixel value to class id

        Returns:
            label_map (np.ndarray): A 2D array (H, W) of class indices.
        """
        # Convert the 3-channel RGB image into a 24-bit integer per pixel.
        # Each pixel's integer is computed as (R << 16) | (G << 8) | B.
        pixel_values = (
            semantic_image[..., 0].astype(np.uint32) << 16
            | semantic_image[..., 1].astype(np.uint32) << 8
            | semantic_image[..., 2].astype(np.uint32)
        )

        # Vectorized lookup using the LUT.
        label_map = self.lut[pixel_values]
        return label_map

    @property
    def num_classes(self) -> int:
        """Number of semantic classes.

        Returns:
            int: Class count.
        """
        return len(self.class_id_to_name)

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.image_paths)

    def _sample_images_from_random_direction(
        self,
        rgb_spherical_image: BatchSphericalImage,
        semantic_spherical_image: BatchSphericalImage,
        random_direction: bool = True,
    ) -> tuple[BatchSphericalImage, BatchSphericalImage]:
        """Sample spherical images from 1/14 random directions using the precomputed rotation matrices.

        Args:
            rgb_spherical_image (BatchSphericalImage): Input spherical image
            semantic_spherical_image (BatchSphericalImage): Corresponding semantic spherical image
            random_direction (bool, optional): Whether to randomly select one of the 14 directions. Defaults to True.

        Returns:
            tuple[BatchSphericalImage, BatchSphericalImage]: sample batch spherical images
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
        else:
            # Use the output_vector as is
            face_vector = output_vector_flat

        # for RGB images
        rgb_sampled_images = self.color_resampler(
            rgb_spherical_image,
            face_vector,
        )
        # for class_maps
        semantic_sampled_images = self.label_resampler(
            semantic_spherical_image,
            face_vector,
        )

        # set the vector to output_vector_flat
        rgb_sampled_images.vector = output_vector_flat
        semantic_sampled_images.vector = output_vector_flat

        return rgb_sampled_images, semantic_sampled_images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieves the sample at the given index.

        Args:
            idx (int): index

        Accesses:
            self.image_paths

        Returns:
            dict[str, Any]: output dict
        """
        rgb_path = self.image_paths[idx]
        semantic_path = rgb_path.replace("/rgb/", "/semantic/", 1).replace(
            "rgb.png", "semantic.png", 1
        )

        # read files
        image = read_file(rgb_path)
        semantic = read_file(semantic_path)

        # mask out border pixels
        black_border_mask = (image[..., :3] != 0).any(axis=-1)
        semantic[~black_border_mask, :3] = 0  # set black border to <UNK> class

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
        )[
            ..., :3
        ]  # shape (H, W, 3)
        semantic = cv2.resize(
            semantic,
            self.downsample_image_size,
            interpolation=cv2.INTER_NEAREST_EXACT,
        )

        # apply possible random reflection
        if random_horizontal_reflection:
            image = np.ascontiguousarray(image[:, ::-1, :])
            semantic = np.ascontiguousarray(semantic[:, ::-1, :])

        if random_vertical_reflection:
            image = np.ascontiguousarray(image[::-1, :, :])
            semantic = np.ascontiguousarray(semantic[::-1, :, :])

        image_tensor = to_torch(image, dtype=torch.float32).unsqueeze(
            0
        )  # shape (1, H, W, 3)
        class_map = self.semantic_to_class_map(semantic)  # shape (H, W)
        # class_map[~self.black_border_mask] = 0  # set black border to <UNK> class
        class_map_tensor = (
            to_torch(class_map, dtype=torch.float32).round().unsqueeze(0).unsqueeze(-1)
        )  # shape (1, H, W, 1)

        # apply possible random erase
        if random_erase:
            image_chw = image_tensor.permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
            class_chw = class_map_tensor.permute(
                0, 3, 1, 2
            ).contiguous()  # (B, 1, H, W)

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
            class_chw = F.erase(
                class_chw,
                *random_erase_parameters[:-1],
                0,
            )  # 1 channel

            image_tensor = image_chw.permute(0, 2, 3, 1).contiguous()  # back to NHWC
            class_map_tensor = class_chw.permute(0, 2, 3, 1).contiguous()

        spherical_image = self._equirectangular_projection(image_tensor[0])
        spherical_class_map = self._equirectangular_projection(class_map_tensor[0])

        spherical_images = BatchSphericalImage(spherical_image)
        spherical_class_maps = BatchSphericalImage(spherical_class_map)

        output_vector_mask = torch.ones(
            image_tensor.shape[1:3], dtype=torch.bool
        )  # as of now all pixels are valid since input is still panoramic

        panoramic_width, panoramic_height = self.downsample_image_size

        # apply possible random rotation
        if random_rotation:
            with fixed_seed(sample_seed):
                R = sample_random_rotation_matrices()

            panoramic_vector = spherical_images.vector.clone()
            R = R.squeeze().to(
                dtype=panoramic_vector.dtype, device=panoramic_vector.device
            )

            # rotate spherical_images
            rotated_panoramic_vector = (R @ panoramic_vector.T).T

            spherical_images = spherical_images.clone()  # avoid in-place modification
            spherical_images.vector = rotated_panoramic_vector

            spherical_class_maps = (
                spherical_class_maps.clone()
            )  # avoid in-place modification
            spherical_class_maps.vector = rotated_panoramic_vector

            if getattr(self, "output_vector", None) is None:
                # resample into panoramic images
                with ephemeral_cache():
                    # ! not using ephemeral=(seed is None) because accumulated cache will be too large, bigger VRAM may help
                    # turn off cache if seed is None because it's true random rotation and
                    # we are unlikely to encounter the same input vector again
                    spherical_images = self.color_resampler(
                        spherical_images,
                        panoramic_vector,
                    )
                    spherical_class_maps = self.label_resampler(
                        spherical_class_maps,
                        panoramic_vector,
                    )

                spherical_class_maps.batch_value = (
                    spherical_class_maps.batch_value.round()
                )

                # extract planar image tensor
                image_tensor = spherical_images.batch_value.clone().reshape(
                    1, panoramic_height, panoramic_width, 3
                )
                class_map_tensor = spherical_class_maps.batch_value.clone().reshape(
                    1, panoramic_height, panoramic_width, 1
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
                    spherical_images, spherical_class_maps = (
                        self._sample_images_from_random_direction(
                            spherical_images,
                            spherical_class_maps,
                            random_direction=self.random_rotation_probability == 0.0,
                            # if not random rotation then we randomly sample from 1/14 directions
                        )
                    )

            spherical_class_maps.batch_value = spherical_class_maps.batch_value.round()

            # extract planar image tensor
            image_tensor = torch.zeros(
                (1, output_image_height, output_image_width, 3),
            )
            image_tensor[:, output_vector_mask] = (
                spherical_images.batch_value.clone().reshape(1, -1, 3)
            )
            class_map_tensor = torch.zeros(
                1,
                output_image_height,
                output_image_width,
                1,
            )  # 0 is the safe default <UNK> class
            class_map_tensor[:, output_vector_mask] = (
                spherical_class_maps.batch_value.clone().reshape(1, -1, 1)
            )

        # enforce default all true mask for safety
        spherical_images.mask = torch.ones(
            spherical_images.vector.shape[0], dtype=torch.bool
        )
        spherical_class_maps.mask = spherical_images.mask

        # valid pixel mask for preprocessing value normalization
        valid_pixel_mask = class_map_tensor[..., 0] != 0  # shape (1, H, W)
        spherical_valid_pixel_mask = BatchSphericalImage(
            batch_value=spherical_class_maps.batch_value[..., :]
            != 0,  # shape (1, N, 1)
            vector=spherical_class_maps.vector,
            mask=spherical_class_maps.mask,
        )

        return {
            "inputs": {
                "images": image_tensor,  # shape (1, H, W, 3)
                "valid_pixel_masks": valid_pixel_mask,  # shape (1, H, W), different across samples because of potential random erase
                "spherical_images": spherical_images,
                "spherical_valid_pixel_masks": spherical_valid_pixel_mask,  # different across samples because of potential random erase
            },
            "labels": {
                "class_maps": class_map_tensor,  # shape (1, H, W, 1)
                "spherical_class_maps": spherical_class_maps,
            },
            "meta": {
                **self.meta,
                "output_vector_mask": output_vector_mask,  # shape (H, W), same across samples
            },
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Custom collate function for PyTorch's DataLoader

        Args:
            batch (list[dict]): A list of samples, where each sample is a dict as returned by __getitem__.

        Returns:
            dict: output dict
        """
        images = torch.cat([sample["inputs"]["images"] for sample in batch], dim=0)
        valid_pixel_masks = torch.cat(
            [sample["inputs"]["valid_pixel_masks"] for sample in batch], dim=0
        )
        class_maps = torch.cat(
            [sample["labels"]["class_maps"] for sample in batch], dim=0
        )

        spherical_images = BatchSphericalImage(
            batch_value=[sample["inputs"]["spherical_images"] for sample in batch],
        )
        spherical_valid_pixel_masks = BatchSphericalImage(
            batch_value=[
                sample["inputs"]["spherical_valid_pixel_masks"] for sample in batch
            ],
        )
        spherical_class_maps = BatchSphericalImage(
            batch_value=[sample["labels"]["spherical_class_maps"] for sample in batch],
        )

        return {
            "inputs": {
                "images": images,
                "valid_pixel_masks": valid_pixel_masks,
                "spherical_images": spherical_images,
                "spherical_valid_pixel_masks": spherical_valid_pixel_masks,
            },
            "labels": {
                "class_maps": class_maps,
                "spherical_class_maps": spherical_class_maps,
            },
            "meta": batch[0]["meta"],
        }

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

    def visualize_batch(
        self,
        batch: dict,
        num_vis: int = -1,
        alpha: float = 0.25,
    ) -> tuple[list[plt.Figure], BatchSphericalImage]:
        """
        Visualize a batch of semantic segmentation predictions.

        Args:
            batch (dict): A batch dict returned from the dataloader or post_collate_processing.
            num_vis (int, optional): Number of samples to visualize. Defaults to -1.
            alpha (float, optional): Alpha blending factor. Defaults to 0.6.

        Returns:
            tuple[list[plt.Figure], BatchSphericalImage]:
                - list of matplotlib figures (planar visualization),
                - BatchSphericalImage.
        """
        num_vis = (
            np.clip(num_vis, 1, batch["inputs"]["images"].shape[0])
            if num_vis != -1
            else num_vis
        )

        images = to_numpy(
            copy_or_clone(
                batch["inputs"]["images"]
                if num_vis == -1
                else batch["inputs"]["images"][:num_vis]
            )
        )  # (B, H, W, 3)
        class_maps = to_numpy(
            copy_or_clone(
                batch["labels"]["class_maps"]
                if num_vis == -1
                else batch["labels"]["class_maps"][:num_vis]
            )
        )  # (B, H, W, num_classes), (B, H, W), or (B, H, W, 1)
        vector = to_numpy(batch["inputs"]["spherical_images"].vector)

        # convert possible one-hot class maps to class maps
        if class_maps.ndim == 4:
            if class_maps.shape[-1] > 1:
                class_maps = class_maps.argmax(axis=-1)  # shape (B, H, W)
            else:
                class_maps = class_maps.squeeze(axis=-1)

        # Create category colormap
        colormap = self._get_category_colormap(self.num_classes)

        blended_images = []
        figures = []
        for img, class_map in zip(images, class_maps):
            overlay = np.zeros_like(img, dtype=np.float32)
            # class_map = class_map.argmax(axis=-1)  # shape (H, W)

            for class_id in np.unique(class_map):
                mask = class_map == class_id
                color = np.array(colormap[int(class_id)][:3]) * 255  # drop alpha
                overlay[mask] = color

            # Alpha blend with input image
            blended = (alpha * overlay + (1 - alpha) * img.astype(np.float32)).astype(
                np.uint8
            )
            blended_images.append(blended)

            # --- Planar Visualization (matplotlib) ---
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(blended)
            ax.axis("off")

            # Add legend
            legend_handles = [
                patches.Patch(color=colormap[i], label=self.class_id_to_name[i])
                for i in np.unique(class_map)
            ]
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.1),
                    ncol=min(5, len(legend_handles)),
                    fontsize="small",
                    frameon=False,
                )

            figures.append(fig)

        return figures, BatchSphericalImage(
            batch_value=np.array(blended_images)[
                :, to_numpy(batch["meta"]["output_vector_mask"]).astype(np.bool_)
            ],
            vector=vector,
            mask=to_numpy(batch["inputs"]["spherical_images"].mask),
        )


class Stanford2D3DSDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Stanford 2D-3D-S semantic segmentation.

    Wraps ``Stanford2D3DSDataset`` for training and validation with distributed
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
        """Initialize the Stanford 2D-3D-S data module.

        Args:
            dataset_base_path (str): Root directory of the 2D-3D-S dataset.
            downsample_image_size (tuple): Target (width, height).
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of DataLoader workers.
            train_augmentation (dict[str, float] | None, optional): Augmentation config for training. Defaults to None.
            val_augmentation (dict[str, float] | None, optional): Augmentation config for validation. Defaults to None.
            train_output_vector (np.ndarray | None, optional): Output grid for training. Defaults to None.
            train_output_vector_mask (np.ndarray | None, optional): Mask for training grid. Defaults to None.
            val_output_vector (np.ndarray | None, optional): Output grid for validation. Defaults to None.
            val_output_vector_mask (np.ndarray | None, optional): Mask for validation grid. Defaults to None.
            val_seed (str | None, optional): Deterministic seed for validation. Defaults to None.
            meta (dict | None, optional): Meta dictionary. Defaults to None.
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
        """No-op: 2D-3D-S data files are expected to already be on disk."""

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train and validation datasets if not already set.

        Args:
            stage (str | None, optional): Lightning stage string. Unused. Defaults to None.
        """
        if (
            getattr(self, "train_dataset", None) is None
        ):  # Avoid re-initialization if already set by x @ Stanford2D3DSDataset
            self.train_dataset = Stanford2D3DSDataset(
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
        ):  # Avoid re-initialization if already set by x @ Stanford2D3DSDataset
            self.val_dataset = Stanford2D3DSDataset(
                dataset_base_path=self.dataset_base_path,
                dataset_type="test",
                augmentation=self.val_augmentation,
                output_vector=self.val_output_vector,
                output_vector_mask=self.val_output_vector_mask,
                downsample_image_size=self.downsample_image_size,
                meta=self.meta,
                seed=self.val_seed,
            )

    def __rmatmul__(self, other: int | float) -> Stanford2D3DSDataModule:
        """Supports x @ Stanford2D3DSDataModule, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Returns:
            Stanford2D3DSDataModule
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
