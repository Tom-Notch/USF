#!/usr/bin/env python3
#
# Created on Sun Feb 09 2025 21:50:35
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from usf.network.layer.spherical.activation import SphericalActivation
from usf.network.layer.spherical.circle_pool import CirclePool
from usf.network.layer.spherical.generic_spherical_cnn import GenericSphericalConv
from usf.network.layer.spherical.global_pool import SphericalGlobalPool
from usf.sampler.sampler import SphericalSampler
from usf.utils.image import fig_to_numpy
from usf.utils.spherical_image import BatchSphericalImage
from usf.utils.torch_numpy import copy_or_clone


class SphericalMnist(nn.Module):
    """Spherical CNN for MNIST classification: 2 x (SphericalConv + ReLU + Pool) -> FC -> 10 classes."""

    def __init__(
        self,
        location_sampler: str,
        backend: str,
        weighting_function_config: dict[str, Any],
        resolution_factor: float,
        reject_oo_fov_vector: bool,
        resampler_config: dict[str, Any] | None = None,
        block_size: int = 8192,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the spherical MNIST model.

        Args:
            location_sampler (str): Tessellation backend name.
            backend (str): Spherical conv backend name.
            weighting_function_config (dict[str, Any]): Config for the conv weighting function.
            resolution_factor (float): Spatial resolution scaling factor.
            reject_oo_fov_vector (bool): Whether to mask out-of-FoV vectors.
            resampler_config (dict[str, Any] | None, optional): Optional input resampler config. Defaults to None.
            block_size (int, optional): Block size for pairwise distance computation. Defaults to 8192.
        """
        super().__init__()

        if resampler_config is not None:
            resampler_config["reject_oo_fov_vector"] = reject_oo_fov_vector
            self.input_resampler = SphericalSampler(resampler_config)

        # 2 x (Convolution + ReLU + Pooling)
        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        GenericSphericalConv(
                            backend=backend,
                            in_channels=1,
                            out_channels=64,
                            radius=torch.pi * 7 / 75,
                            weighting_function_config=weighting_function_config,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "act1",
                        SphericalActivation("ReLU", inplace=True),
                    ),
                    (
                        "pool1",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 0.045,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "conv2",
                        GenericSphericalConv(
                            backend=backend,
                            in_channels=64,
                            out_channels=256,
                            radius=torch.pi * 7 / 30,
                            weighting_function_config=weighting_function_config,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    ("act2", SphericalActivation("ReLU", inplace=True)),
                    (
                        "pool2",
                        SphericalGlobalPool(pool_type="max"),
                    ),
                ]
            )
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: BatchSphericalImage) -> torch.Tensor:
        """Main forward function

        Args:
            x (BatchSphericalImage): Input spherical image.

        Returns:
            torch.Tensor: shape (B, 10)
        """
        if getattr(self, "input_resampler", None) is not None:
            x = self.input_resampler(x)

        x = self.conv_block(x)

        batch_value = x.batch_value[:, 0, :]  # (B, N, C) -> (B, C)
        batch_value = F.relu(self.fc1(batch_value))

        batch_value = self.fc2(batch_value)
        return batch_value


class PlanarMnist(nn.Module):
    """Planar CNN baseline for MNIST: 2 x (Conv2d + ReLU + Pool) -> FC -> 10 classes."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the planar MNIST model."""
        super().__init__()

        # 2 x (Convolution + ReLU + Pooling)
        self.conv_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=1, out_channels=64, kernel_size=3, padding=1
                        ),
                    ),
                    (
                        "act1",
                        nn.ReLU(inplace=True),
                    ),
                    ("pool1", nn.MaxPool2d(kernel_size=2)),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=64, out_channels=256, kernel_size=3, padding=1
                        ),
                    ),
                    ("act2", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Main forward function

        Args:
            x (torch.Tensor): expect in format (B, C, H, W)

        Returns:
            torch.Tensor: shape (B, 10)
        """
        batch_size = x.shape[0]

        x = self.conv_block(x)
        x = torch.amax(x, dim=(-2, -1))

        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x


class MNISTLightningModel(pl.LightningModule):
    """Lightning wrapper for MNIST classification, dispatching to Planar or Spherical backbone.

    Logs cross-entropy loss and accuracy, and visualizes predictions on the first
    validation batch each epoch.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        visualization_config: dict,
        log_image_dpi: int,
        architecture: PlanarMnist | SphericalMnist,
    ) -> None:
        """Initialize the MNIST Lightning model.

        Args:
            lr (float): Learning rate for AdamW.
            weight_decay (float): Weight decay for AdamW.
            visualization_config (dict): Kwargs passed to ``MNISTDataset.visualize_batch``.
            log_image_dpi (int): DPI for logged visualizations.
            architecture (PlanarMnist | SphericalMnist): The backbone model.
        """
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.visualization_config = visualization_config
        self.log_image_dpi = log_image_dpi

        self.model = architecture

        for param in self.parameters():
            param.data = param.data.contiguous()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load state dict with forced ``strict=False`` and auto-register unexpected buffers.

        Buffers created by cached samplers/layers may not exist at ``__init__`` time
        but are present in the checkpoint. This method registers them automatically.

        Args:
            state_dict (Mapping[str, Any]): State dictionary from a checkpoint.
            strict (bool, optional): Ignored — always forced to False. Defaults to True.
            assign (bool, optional): Whether to assign tensors directly. Defaults to False.

        Returns:
            Any: ``_IncompatibleKeys`` named tuple from ``nn.Module.load_state_dict``.
        """
        # ! forcefully set strict to False because some buffers are not declared in the model when init but exist in the state dict, this is not a bug
        strict = False
        # always allow extras
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        # for every key that was in the checkpoint but not found on self,
        # try to register it as a buffer on the correct submodule
        total_registered_buffers = 0
        for full_name in result.unexpected_keys:
            tensor = state_dict[full_name]
            if not isinstance(tensor, torch.Tensor):
                # only buffers are tensors
                continue

            # walk down the name path to find the parent module
            parts = full_name.split(".")
            parent = self
            for p in parts[:-1]:
                parent = getattr(parent, p)

            # register the buffer
            buf_name = parts[-1]
            if not hasattr(parent, buf_name):
                parent.register_buffer(buf_name, tensor)
                print(f"  + registered buffer {full_name}: shape {tuple(tensor.shape)}")
                total_registered_buffers += 1

        print(
            f"✔ state dict loaded: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected."
        )

        if len(result.unexpected_keys) > 0:
            print(
                f"✔ registered {total_registered_buffers} unexpected keys from state dict as buffers."
            )
        # now that we've registered them, we could even re-call load_state_dict
        # to get them into self.buffers, but it's usually fine to stop here.

        return result

    def forward(self, inputs: dict) -> torch.Tensor:
        """Dispatch to Planar or Spherical backbone.

        Args:
            inputs (dict): Dictionary with ``"images"`` and/or ``"spherical_images"``.

        Returns:
            torch.Tensor: Raw logits of shape (B, 10).

        Raises:
            ValueError: If the backbone type is unsupported.
        """
        if isinstance(self.model, PlanarMnist):
            images = inputs["images"].permute(0, 3, 1, 2)  # Convert NHWC -> NCHW
            return self.model(images)
        elif isinstance(self.model, SphericalMnist):
            return self.model(inputs["spherical_images"])
        else:
            raise ValueError(
                f"Unsupported model type: {type(self.model)}. "
                "Expected PlanarMnist or SphericalMnist."
            )

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute cross-entropy loss and log loss + accuracy.

        Args:
            batch (dict): Batch from dataloader with ``"inputs"`` and ``"labels"``.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Scalar training loss.
        """
        outputs = self(batch["inputs"])
        labels = batch["labels"].long()
        loss = F.cross_entropy(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()

        self.log_dict(
            {
                "train loss": loss,
                "train accuracy": accuracy,
            },
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=labels.shape[0],
        )

        return loss

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute validation loss/accuracy and log prediction visualization on batch 0.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Scalar validation loss.
        """
        outputs = self(batch["inputs"])
        labels = batch["labels"].long()
        loss = F.cross_entropy(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()

        self.log_dict(
            {
                "validation loss": loss,
                "validation accuracy": accuracy,
            },
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=labels.shape[0],
        )

        if batch_idx == 0 and self.logger is not None:
            vis_batch = batch.copy()
            vis_batch["labels"] = copy_or_clone(outputs).argmax(dim=-1)

            fig = self.trainer.val_dataloaders.dataset.visualize_batch(
                vis_batch, **self.visualization_config
            )
            image = fig_to_numpy(fig, dpi=self.log_image_dpi)
            plt.close(fig)
            self.log_images(
                images=[image],
                captions=[f"Epoch {self.trainer.current_epoch}"],
                key="Validation Prediction",
            )

        return loss

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Store raw logit outputs in the batch dict and return it.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            dict: Updated batch with ``"outputs"`` key.
        """
        batch["outputs"] = self(batch["inputs"])
        return batch

    def log_images(
        self,
        images: list[np.ndarray],
        captions: list[str],
        key: str,
    ) -> None:
        """log images to all loggers

        Args:
            images (list[np.ndarray]): list of RGBA numpy images
            captions (list[str]): list of captions for each image
            key (str): key to log the images under
        """
        for logger in getattr(self.trainer, "loggers", []):
            if hasattr(logger, "log_image"):
                # Wandb, Comet, Neptune
                logger.log_image(
                    key=key,
                    images=images,  # support numpy HWC(A)
                    caption=captions,
                    step=self.global_step,
                )
            elif isinstance(logger, pl.loggers.TensorBoardLogger):
                for i, img in enumerate(images):
                    tag = f"{key}/{captions[i]}"
                    logger.experiment.add_image(
                        tag=tag,
                        img_tensor=img[..., :3].transpose(2, 0, 1),  # HWC → CHW
                        global_step=self.global_step,
                        dataformats="CHW",
                    )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer.

        Returns:
            torch.optim.Optimizer: AdamW with configured lr and weight_decay.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
