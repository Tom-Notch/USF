#!/usr/bin/env python3
#
# Created on Sat Feb 08 2025 17:46:17
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import math
import struct
import warnings
from array import array
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from usf.generate_lens_normal_map import generate_stereographic_normal_map
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.cache import ephemeral_cache
from usf.utils.files import parse_path
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import fixed_seed, string_to_seed, to_torch


class MNISTDataset(Dataset):
    """Spherical MNIST dataset with optional stereographic projection and random rotation augmentation.

    Each grayscale MNIST image is projected onto the sphere using a stereographic
    projection, yielding a ``SphericalImage`` with per-pixel unit vectors. Optional
    random rotation augmentation spins the sphere about the x-axis (YZ-plane rotation)
    and resamples pixel values via RBF interpolation to simulate viewpoint invariance.

    The ``@`` operator (``n @ dataset``) subsamples the dataset to *n* samples (int)
    or a fraction of the full set (float), returning a deep copy.
    """

    def __init__(
        self,
        dataset_base_path: str,
        dataset_type: str,
        augmentation: dict[str, float] | None = None,
        seed: str | None = None,
    ):
        """Initialize the MNIST dataset.

        Args:
            dataset_base_path (str): Base directory where the MNIST binary files are stored.
            dataset_type (str): Which split to load — ``'train'`` or ``'test'``.
            augmentation (dict[str, float] | None, optional): Augmentation probabilities keyed
                by name. Currently supports ``'rotation'`` (probability of applying a random
                YZ-plane rotation). Defaults to None (no augmentation).
            seed (str | None, optional): Deterministic seed string for reproducible augmentation.
                When None, augmentation is random each call. Defaults to None.

        Raises:
            ValueError: If ``dataset_type`` is not ``'train'`` or ``'test'``.
        """
        self.dataset_base_path = parse_path(dataset_base_path)
        assert self.dataset_base_path, f"Invalid dataset_base_path {dataset_base_path}"

        # --- Data Augmentation ---
        augmentation = augmentation or {}  # default to empty dict

        # rotation
        self.random_rotation_probability = augmentation.get("rotation", 0.0)
        if (
            self.random_rotation_probability > 0.0
            and getattr(self, "color_resampler", None) is None
        ):
            self.color_resampler = ValueSampler(
                {
                    "value_sampler": "radial_basis_function",
                    "value_sampler_config": {
                        "radius": 0.1000586450099945,
                        # "num_in_circle_points": 4,
                        "kernel": "gaussian",
                        "sigma": 0.2,
                    },
                    "reject_oo_fov_value": False,
                }
            )

        # Build file paths based on the dataset type.
        if dataset_type == "train":
            self.images_path = parse_path(
                self.dataset_base_path + "/train-images.idx3-ubyte"
            )
            self.labels_path = parse_path(
                self.dataset_base_path + "/train-labels.idx1-ubyte"
            )
        elif dataset_type == "test":
            self.images_path = parse_path(
                self.dataset_base_path + "/t10k-images.idx3-ubyte"
            )
            self.labels_path = parse_path(
                self.dataset_base_path + "/t10k-labels.idx1-ubyte"
            )
        else:
            raise ValueError(
                f"dataset_type {dataset_type} must be either 'train' or 'test'"
            )

        assert (
            self.images_path and self.labels_path
        ), "MNIST subfolder structure not following canonical"

        # Load images and labels into memory.
        self.images, self.labels = self.read_images_labels(
            self.images_path, self.labels_path
        )

        self.seed = seed

    def __rmatmul__(self, other: int | float) -> MNISTDataset:
        """Supports x @ MNISTDataset, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Raises:
            TypeError: If `other` is not an int or float.

        Returns:
            MNISTDataset
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
        dataset_copy.images = [self.images[i] for i in random_indices]
        dataset_copy.labels = [self.labels[i] for i in random_indices]

        return dataset_copy

    def read_images_labels(
        self, images_path: str, labels_path: str
    ) -> tuple[list[np.ndarray], array]:
        """Read MNIST IDX binary files and return images and labels.

        Args:
            images_path (str): Path to the IDX3-ubyte images file.
            labels_path (str): Path to the IDX1-ubyte labels file.

        Returns:
            tuple[list[np.ndarray], array]: A 2-tuple of:
                - images: list of float32 arrays each of shape (H, W).
                - labels: array of uint8 class labels, length N.

        Raises:
            ValueError: If the magic number in either binary file does not match the IDX format spec.
        """
        # Load labels.
        with open(labels_path, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Magic number mismatch in label file, expected 2049, got {magic}"
                )
            labels = array("B", file.read())

        # Load images.
        with open(images_path, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch in image file, expected 2051, got {magic}"
                )
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            start = i * rows * cols
            end = start + rows * cols
            # Convert each image into a 2D numpy array.
            img = np.array(image_data[start:end], dtype=np.float32).reshape(rows, cols)
            images.append(img)

        return images, labels

    def _stereographic_projection(
        self, image: np.ndarray | torch.Tensor
    ) -> BatchSphericalImage:
        """
        Internal method to compute a SphericalImage from a given image using stereographic projection.
        It calls on generate_stereographic_normal_map to compute the per-pixel unit vectors.

        Args:
            image (np.ndarray | torch.Tensor): Input image with shape (H, W, C).

        Returns:
            BatchSphericalImage: An instance containing:
                - value: flattened image data of shape (H*W, C)
                - vector: flattened lens normal map of shape (H*W, 3)
                - polar: left as None (to be computed in SphericalImage.__post_init__)
                - path: None
        """
        # Convert to NumPy if needed.
        if isinstance(image, np.ndarray):
            _image = to_torch(image)

        H, W, C = _image.shape
        # Warn if the _image is not square.
        if H != W:
            warnings.warn(
                "The input image is not square. Stereographic projection may lead to non-uniform distortions.",
                UserWarning,
                stacklevel=2,
            )

        # Use the imported function to generate the per-pixel unit vectors.
        # This returns a lens normal map of shape (H, W, 3).
        vector = to_torch(
            generate_stereographic_normal_map(
                H, W, projection_point=(-1, 0, 0), up=(0, 0, 1)
            ),
        )

        # Create and return the SphericalImage instance.
        return BatchSphericalImage(
            batch_value=_image.view(1, -1, C), vector=vector.view(-1, 3)
        )

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.labels)

    @staticmethod
    def _sample_random_YZ_rotation() -> torch.Tensor:
        """
        Sample a rotate in YZ plane

        Returns:
            torch.Tensor: (3, 3) rotation matrix R_x(α), α ~ U[0, 2π).
        """
        alpha = 2 * math.pi * torch.rand(())  # roll angle
        cos, sin = torch.cos(alpha), torch.sin(alpha)

        # R_x(α) = [[1, 0,  0],
        #           [0,  c, -s],
        #           [0,  s,  c]]
        zero = torch.zeros_like(cos)
        one = torch.ones_like(cos)
        return torch.stack(
            [
                torch.stack([one, zero, zero]),
                torch.stack([zero, cos, -sin]),
                torch.stack([zero, sin, cos]),
            ]
        )

    def __getitem__(self, idx: int) -> dict:
        """Retrieve the sample at the given index.

        Args:
            idx (int): Index into the dataset.

        Returns:
            dict: A dictionary with keys:
                - "inputs": a dictionary with keys:
                    - "images": a tensor of shape (1, rows, cols, C).
                    - "labels": a scalar tensor.
                - "spherical_images": a BatchSphericalImage computed via the internal _stereographic_projection method.
        """
        image = self.images[idx]

        # Ensure the image has a channel dimension
        if image.ndim == 2:
            image = image[..., None]

        label = self.labels[idx]
        image_tensor = torch.tensor(
            image,
        )
        label_tensor = torch.tensor(label)

        # Compute the spherical image using the internal method.
        batch_spherical_image = self._stereographic_projection(image)

        # determine seed for reproducibility
        sample_seed = (
            None if self.seed is None else string_to_seed(self.seed + f" {idx}")
        )

        with fixed_seed(sample_seed):
            random_rotation = bool(
                (torch.rand(()) < self.random_rotation_probability).item()
            )

        if random_rotation:
            with fixed_seed(sample_seed):
                R = self._sample_random_YZ_rotation()

            original_vector = batch_spherical_image.vector.clone()
            R = R.to(dtype=original_vector.dtype, device=original_vector.device)
            batch_spherical_image.vector = (R @ original_vector.T).T  # rotate

            with ephemeral_cache():
                # ! not using ephemeral=(seed is None) because accumulated cache will be too large, bigger VRAM may help
                batch_spherical_image = self.color_resampler(
                    batch_spherical_image, original_vector
                )

            batch_spherical_image.mask = torch.ones(
                batch_spherical_image.vector.shape[0], dtype=torch.bool
            )

            image_tensor = batch_spherical_image.batch_value.clone().reshape(
                image_tensor.shape
            )

        return {
            "inputs": {
                "images": image_tensor.unsqueeze(0),
                "spherical_images": batch_spherical_image,
            },
            "labels": label_tensor.unsqueeze(0),
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        """
        Custom collate function for PyTorch's DataLoader.

        Args:
            batch (list): A list of samples, where each sample is a dict as returned by __getitem__.

        Returns:
            dict: A dictionary with keys:
                - "inputs": a dictionary with keys:
                    - "images": a tensor of shape (batch_size, rows, cols, C) created by stacking image tensors.
                    - "labels": a tensor of shape (batch_size,) created by stacking label tensors.
                - "batch_spherical_image": a BatchSphericalImage instance built from the list of spherical images.
        """
        images = torch.cat([sample["inputs"]["images"] for sample in batch], dim=0)
        labels = torch.cat([sample["labels"] for sample in batch], dim=0)

        # Build the BatchSphericalImage from the list of per-sample spherical images.
        # Assume that all spherical images share the same geometry (vector and polar).
        batch_spherical_image = BatchSphericalImage(
            batch_value=[sample["inputs"]["spherical_images"] for sample in batch],
        )

        return {
            "inputs": {
                "images": images,
                "spherical_images": batch_spherical_image,
            },
            "labels": labels,
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

    @staticmethod
    def visualize_batch(
        batch: dict, num_vis: int = 32, num_columns: int = 8
    ) -> plt.Figure:
        """
        Visualize a batch from the dataloader.

        This function takes a batch (a dict with keys "images" and "labels") and arranges the images
        in a grid with the corresponding label underneath each image. It does not assume that the tensors
        are on CPU (they are cloned and moved to CPU for visualization), and it produces a matplotlib Figure
        without side effects (i.e. it does not call plt.show()).

        It now supports images that have an extra channel dimension. If the channel dimension is 1,
        the image is squeezed to (H, W) and displayed in grayscale. Otherwise (e.g. if C==3), the image
        is displayed in color.

        Args:
            batch (dict): A dictionary with keys:
                - "inputs": a dictionary with keys:
                    - "images": a tensor of shape (batch_size, H, W, C) or (batch_size, H, W) if already squeezed.
                    - "spherical_images": a BatchSphericalImage
                - "labels": a tensor of shape (batch_size,)
            num_vis (int): Maximum number of images to visualize. Default is 32.
            num_columns (int): Number of columns in the grid. Default is 8.

        Returns:
            matplotlib.Figure: The composite figure containing the visualized batch.
        """
        num_vis = np.clip(num_vis, 1, batch["inputs"]["images"].shape[0])

        # Clone and move to CPU (without modifying the original tensors)
        images = batch["inputs"]["images"].clone().cpu()[:num_vis, ...]
        labels = batch["labels"].clone().cpu()[:num_vis, ...]

        # If images have an extra channel dimension (H, W, C) with C==1, squeeze it.
        if images.ndim == 4 and images.shape[-1] == 1:
            images = images.squeeze(-1)  # Now shape becomes (B, H, W)

        # Compute grid dimensions.
        num_rows = math.ceil(num_vis / num_columns)
        fig, axes = plt.subplots(
            num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2)
        )

        # Ensure axes is a 2D array.
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if num_columns == 1:
            axes = np.expand_dims(axes, axis=1)

        # Plot each image.
        for idx in range(num_vis):
            row = idx // num_columns
            col = idx % num_columns
            ax = axes[row, col]
            img = images[idx]
            # If image is 2D, assume grayscale; if it is 3D, assume color.
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(int(labels[idx].item()), fontsize=10, labelpad=5)

        # Turn off any unused subplots.
        total_subplots = num_rows * num_columns
        for idx in range(num_vis, total_subplots):
            row = idx // num_columns
            col = idx % num_columns
            axes[row, col].axis("off")

        fig.tight_layout()
        return fig


class MNISTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for spherical MNIST.

    Wraps ``MNISTDataset`` for training and validation, handling distributed
    sampling automatically when ``torch.distributed`` is initialized. Supports
    the ``@`` operator (``n @ datamodule``) to sub-sample both splits for fast
    overfit experiments.
    """

    def __init__(
        self,
        dataset_base_path: str,
        batch_size: int,
        num_workers: int,
        train_augmentation: dict[str, float] | None = None,
        val_augmentation: dict[str, float] | None = None,
        val_seed: str | None = None,
    ):
        """Initialize the MNIST data module.

        Args:
            dataset_base_path (str): Root directory containing the MNIST binary files.
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of DataLoader worker processes.
            train_augmentation (dict[str, float] | None, optional): Augmentation config
                for the training split. Defaults to None.
            val_augmentation (dict[str, float] | None, optional): Augmentation config
                for the validation split. Defaults to None.
            val_seed (str | None, optional): Deterministic seed string for reproducible
                validation augmentation. Defaults to None.
        """
        super().__init__()
        self.dataset_base_path = dataset_base_path

        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation
        self.val_seed = val_seed

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """No-op: MNIST binary files are expected to already be on disk."""

    def setup(self, stage: str | None = None) -> None:
        """Instantiate train and validation datasets if not already set.

        Skips re-initialization if datasets were pre-assigned (e.g. via ``@``).

        Args:
            stage (str | None, optional): Lightning stage string (``'fit'``, ``'test'``,
                etc.). Unused; datasets are always initialized. Defaults to None.
        """
        if (
            getattr(self, "train_dataset", None) is None
        ):  # Avoid re-initializing if already set by x @ MNISTDataModule
            self.train_dataset = MNISTDataset(
                dataset_base_path=self.dataset_base_path,
                dataset_type="train",
                augmentation=self.train_augmentation,
            )
        if (
            getattr(self, "val_dataset", None) is None
        ):  # Avoid re-initializing if already set by x @ MNISTDataModule
            self.val_dataset = MNISTDataset(
                dataset_base_path=self.dataset_base_path,
                dataset_type="test",
                augmentation=self.val_augmentation,
                seed=self.val_seed,
            )

    def __rmatmul__(self, other: int | float) -> MNISTDataModule:
        """Supports x @ MNISTDataModule, where x is an int or float.

        Args:
            other (int | float): int for # samples, float for percentage of samples.

        Returns:
            MNISTDataModule
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
