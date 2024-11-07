#!/usr/bin/env python3
#
# Created on Fri Apr 04 2025 22:52:00
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import Conv2d
from torch.utils.checkpoint import checkpoint

from usf.network.block.planar.c3k2 import PlanarC3K2
from usf.network.block.planar.cbna import PlanarCBNA
from usf.network.block.planar.deeplabv3 import PlanarDeepLabV3
from usf.network.block.planar.unet import PlanarUNet
from usf.network.block.planar.yolov11 import PlanarYOLOv11
from usf.network.block.spherical.c3k2 import SphericalC3K2
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.block.spherical.deeplabv3 import SphericalDeepLabV3
from usf.network.block.spherical.unet import SphericalUNet
from usf.network.block.spherical.yolov11 import SphericalYOLOv11
from usf.network.layer.spherical.conv1x1 import SphericalConv1x1
from usf.sampler.sampler import SphericalSampler
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.image import fig_to_numpy
from usf.utils.lr_scheduler import get_cosine_with_min_lr_schedule_with_warmup
from usf.utils.spherical_image import BatchSphericalImage, concatenate
from usf.utils.torch_numpy import copy_or_clone, to_torch


class SphericalSemanticSegmentation(nn.Module):
    """Spherical dense per-pixel semantic segmentation model.

    Supports YOLOv11, UNet, and DeepLabV3 backbone architectures operating on
    ``BatchSphericalImage``. Optional input/output resamplers allow projecting
    between the input sphere tessellation and an internal processing grid.
    """

    BACKBONES = {"YOLOv11", "UNet", "DeepLabV3"}

    def __init__(
        self,
        backbone: str,
        num_categories: int,
        base_channel: int,
        activation: str,
        location_sampler: str,
        expansion: float,
        resolution_factor: float,
        backend: str,
        weighting_function_config: dict,
        reject_oo_fov_vector: bool,
        resampler_config: dict[str, Any] | None = None,
        block_size: int = 8192,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the spherical semantic segmentation model.

        Args:
            backbone (str): One of ``"YOLOv11"``, ``"UNet"``, ``"DeepLabV3"``.
            num_categories (int): Number of output semantic classes.
            base_channel (int): Base channel width multiplied through stages.
            activation (str): Activation function name.
            location_sampler (str): Tessellation backend name.
            expansion (float): Channel expansion factor for blocks.
            resolution_factor (float): Spatial resolution scaling factor.
            backend (str): Spherical conv backend name.
            weighting_function_config (dict): Config for the conv weighting function.
            reject_oo_fov_vector (bool): Whether to mask out-of-FoV vectors.
            resampler_config (dict[str, Any] | None, optional): Optional input resampler config. Defaults to None.
            block_size (int, optional): Block size for pairwise distance computation. Defaults to 8192.
        """
        super().__init__()

        assert (
            backbone in self.BACKBONES
        ), f"backbone not supported, available options: {self.BACKBONES}"

        if resampler_config is not None:
            resampler_config["reject_oo_fov_vector"] = reject_oo_fov_vector
            self.input_resampler = SphericalSampler(resampler_config)
            self.output_resampler = ValueSampler(
                {
                    "value_sampler": "nearest_neighbor",
                    "value_sampler_config": {"num_neighbors": 1},
                    "reject_oo_fov_value": resampler_config["reject_oo_fov_value"],
                }
            )

        if backbone == "YOLOv11":
            self.backbone = SphericalYOLOv11(
                base_channel=base_channel,
                activation=activation,
                location_sampler=location_sampler,
                expansion=expansion,
                resolution_factor=resolution_factor,
                multilevel=False,
                backend=backend,
                weighting_function_config=weighting_function_config,
                reject_oo_fov_vector=reject_oo_fov_vector,
                block_size=block_size,
                *args,
                **kwargs,
            )

            self.upsample3 = SphericalCBNA(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**1,
                # backend=backend,
                # weighting_function_config=weighting_function_config,
                # radius=torch.pi * 1.2 / 40,
                activation=activation,
                # location_sampler=location_sampler,
                # reject_oo_fov_vector=reject_oo_fov_vector,
                # resolution_factor=1 / resolution_factor,
                # block_size=block_size,
                interpolation={
                    "value_sampler": "radial_basis_function",
                    "value_sampler_config": {
                        "radius": 0.054967448115348816,
                        # "num_in_circle_points": 4,
                        "kernel": "gaussian",
                        "sigma": 0.2,
                        "block_size": block_size,
                    },
                },
            )

            self.c3k2_upsample3 = SphericalC3K2(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**1,
                activation=activation,
                expansion=expansion,
                num_c3k=3,
                shortcut=True,
                backend=backend,
                weighting_function_config=weighting_function_config,
                radius=torch.pi * 1.5 / 90,
                block_size=block_size,
            )

            self.upsample4 = SphericalCBNA(
                in_channels=base_channel * 2**1,
                out_channels=base_channel * 2**0,
                # backend=backend,
                # weighting_function_config=weighting_function_config,
                # radius=torch.pi * 0.65 / 40,
                activation=activation,
                # location_sampler=location_sampler,
                # reject_oo_fov_vector=reject_oo_fov_vector,
                # resolution_factor=1 / resolution_factor,
                # block_size=block_size,
                interpolation={
                    "value_sampler": "radial_basis_function",
                    "value_sampler_config": {
                        "radius": 0.024892378598451614,
                        # "num_in_circle_points": 4,
                        "kernel": "gaussian",
                        "sigma": 0.2,
                        "block_size": block_size,
                    },
                },
            )

            self.upsample5 = SphericalCBNA(
                in_channels=base_channel * 2**1,
                out_channels=base_channel * 2**0,
                # backend=backend,
                # weighting_function_config=weighting_function_config,
                # radius=torch.pi * 1.25 / 200,
                activation=activation,
                # location_sampler=location_sampler,
                # reject_oo_fov_vector=reject_oo_fov_vector,
                # resolution_factor=1 / resolution_factor,
                # block_size=block_size,
                interpolation={
                    "value_sampler": "radial_basis_function",
                    "value_sampler_config": {
                        "radius": 0.01183754438534379,
                        # "num_in_circle_points": 4,
                        "kernel": "gaussian",
                        "sigma": 0.2,
                        "block_size": block_size,
                    },
                },
            )

            self.output_projection = SphericalConv1x1(
                in_channels=base_channel,
                out_channels=num_categories,
                bias=True,
            )

        elif backbone == "UNet":
            self.backbone = SphericalUNet(
                base_channel=base_channel,
                activation=activation,
                location_sampler=location_sampler,
                backend=backend,
                weighting_function_config=weighting_function_config,
                resolution_factor=resolution_factor,
                reject_oo_fov_vector=reject_oo_fov_vector,
                block_size=block_size,
                *args,
                **kwargs,
            )
            self.output_projection = SphericalConv1x1(
                in_channels=base_channel,
                out_channels=num_categories,
                bias=True,
            )

        elif backbone == "DeepLabV3":
            self.backbone = SphericalDeepLabV3(
                base_channel=base_channel,
                activation=activation,
                expansion=expansion,
                location_sampler=location_sampler,
                backend=backend,
                weighting_function_config=weighting_function_config,
                resolution_factor=resolution_factor,
                reject_oo_fov_vector=reject_oo_fov_vector,
                block_size=block_size,
                *args,
                **kwargs,
            )
            self.output_projection = SphericalConv1x1(
                in_channels=base_channel * 2**2,
                out_channels=num_categories,
                bias=True,
            )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: optional resample -> backbone -> projection -> optional resample back.

        Args:
            batch_spherical_image (BatchSphericalImage): Input spherical image.

        Returns:
            BatchSphericalImage: Per-pixel class logits.
        """
        # Resample if configured
        preprocessed_batch_spherical_image = (
            batch_spherical_image
            if getattr(self, "input_resampler", None) is None
            else self.input_resampler(batch_spherical_image)
        )

        backbone_output = checkpoint(
            self.backbone,
            preprocessed_batch_spherical_image,
            use_reentrant=False,
        )

        if isinstance(self.backbone, SphericalYOLOv11):
            self.upsample3[0].set_output_vector(backbone_output["down2"].vector)
            up3_batch_spherical_image = self.c3k2_upsample3(
                concatenate(
                    [
                        backbone_output["down2"],
                        self.upsample3(backbone_output["high_resolution"]),
                    ]
                )
            )

            self.upsample4[0].set_output_vector(backbone_output["down1"].vector)
            up4_batch_spherical_image = concatenate(
                [
                    backbone_output["down1"],
                    self.upsample4(up3_batch_spherical_image),
                ]
            )

            self.upsample5[0].set_output_vector(
                preprocessed_batch_spherical_image.vector
            )
            up5_batch_spherical_image = self.upsample5(up4_batch_spherical_image)
            top_batch_spherical_image = self.output_projection(
                up5_batch_spherical_image
            )
        elif isinstance(self.backbone, (SphericalUNet, SphericalDeepLabV3)):
            top_batch_spherical_image = self.output_projection(backbone_output)

        # Resample into inputs pixel distribution
        class_map_batch_spherical_image = (
            top_batch_spherical_image
            if getattr(self, "output_resampler", None) is None
            else self.output_resampler(
                top_batch_spherical_image,
                batch_spherical_image.vector,
            )
        )

        return class_map_batch_spherical_image


class PlanarSemanticSegmentation(nn.Module):
    """Planar semantic segmentation following YOLOv11 architecture."""

    BACKBONES = {"YOLOv11", "UNet", "DeepLabV3"}

    def __init__(
        self,
        backbone: str,
        num_categories: int,
        base_channel: int,
        activation: str,
        expansion: float,
        kernel_size: int,
        *args,
        **kwargs,
    ):
        """Initialize PlanarSemanticSegmentation.

        Args:
            backbone (str): Backbone architecture name, one of {"YOLOv11", "UNet", "DeepLabV3"}.
            num_categories (int): Number of semantic categories.
            base_channel (int): Base number of channels for feature extraction.
            activation (str): Activation function type.
            expansion (float): Expansion ratio for hidden layers.
            kernel_size (int): Kernel size for convolutions (must be odd).
        """
        super().__init__()

        assert (
            backbone in self.BACKBONES
        ), f"backbone not supported, available options: {self.BACKBONES}"

        assert kernel_size % 2 == 1, "Kernel size must be an odd number."

        if backbone == "YOLOv11":
            self.backbone = PlanarYOLOv11(
                base_channel=base_channel,
                activation=activation,
                expansion=expansion,
                multilevel=False,
                kernel_size=kernel_size,
                *args,
                **kwargs,
            )

            self.upsample3 = PlanarCBNA(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**1,
                activation=activation,
                kernel_size=kernel_size,
                stride=2,
                upsample="interpolation",
            )

            self.c3k2_upsample3 = PlanarC3K2(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**1,
                activation=activation,
                kernel_sizes=[kernel_size, kernel_size],
                num_c3k=3,
                expansion=expansion,
                shortcut=True,
            )

            self.upsample4 = PlanarCBNA(
                in_channels=base_channel * 2**1,
                out_channels=base_channel * 2**0,
                activation=activation,
                kernel_size=kernel_size,
                stride=2,
                upsample="interpolation",
            )

            self.upsample5 = PlanarCBNA(
                in_channels=base_channel * 2**1,
                out_channels=base_channel * 2**0,
                activation=activation,
                kernel_size=kernel_size,
                stride=2,
                upsample="interpolation",
            )

            self.output_projection = Conv2d(
                in_channels=base_channel,
                out_channels=num_categories,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        elif backbone == "UNet":
            self.backbone = PlanarUNet(
                base_channel=base_channel,
                activation=activation,
                kernel_size=kernel_size,
                *args,
                **kwargs,
            )
            self.output_projection = Conv2d(
                in_channels=base_channel,
                out_channels=num_categories,
                kernel_size=1,
                stride=1,
                bias=True,
            )
        elif backbone == "DeepLabV3":
            self.backbone = PlanarDeepLabV3(*args, **kwargs)
            self.output_projection = Conv2d(
                in_channels=256,
                out_channels=num_categories,
                kernel_size=1,
                stride=1,
                bias=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the object detection model."""
        backbone_output = self.backbone(x)

        if isinstance(self.backbone, PlanarYOLOv11):
            up3 = self.c3k2_upsample3(
                torch.cat(
                    [
                        backbone_output["down2"],
                        self.upsample3(backbone_output["high_resolution"]),
                    ],
                    dim=1,
                )
            )

            up4 = torch.cat(
                [
                    backbone_output["down1"],
                    self.upsample4(up3),
                ],
                dim=1,
            )

            up5 = self.upsample5(up4)

            output = self.output_projection(up5)
        elif isinstance(self.backbone, (PlanarUNet, PlanarDeepLabV3)):
            output = self.output_projection(backbone_output)

        return output


class SemanticSegmentationLightningModel(pl.LightningModule):
    """Lightning wrapper for semantic segmentation with cross-entropy + Dice loss and mIoU/mAcc benchmarking."""

    def __init__(
        self,
        optimizer_param: dict,
        lr_scheduler_param: dict,
        benchmark_train: bool,
        loss_param: dict[str, Any],
        num_vis: int,
        log_image_dpi: int,
        architecture: PlanarSemanticSegmentation | SphericalSemanticSegmentation,
    ) -> None:
        """Initialize the semantic segmentation Lightning model.

        Args:
            optimizer_param (dict): Kwargs passed to AdamW.
            lr_scheduler_param (dict): Kwargs passed to cosine LR scheduler.
            benchmark_train (bool): Whether to compute mIoU/mAcc during training.
            loss_param (dict[str, Any]): Parameters for the combined loss function.
            num_vis (int): Number of samples to visualize per validation epoch.
            log_image_dpi (int): DPI for logged visualizations.
            architecture (PlanarSemanticSegmentation | SphericalSemanticSegmentation): The backbone model (Planar or Spherical).
        """
        super().__init__()

        self.optimizer_param = optimizer_param
        self.lr_scheduler_param = lr_scheduler_param

        self.benchmark_train = benchmark_train
        self.loss_param = loss_param
        self.num_vis = num_vis
        self.log_image_dpi = log_image_dpi
        self.model = architecture

        for param in self.parameters():
            param.data = param.data.contiguous()

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load state dict with forced ``strict=False`` and auto-register unexpected buffers.

        Args:
            state_dict (Mapping[str, Any]): State dictionary from a checkpoint.
            strict (bool, optional): Ignored — always forced to False. Defaults to True.
            assign (bool, optional): Whether to assign tensors directly. Defaults to False.

        Returns:
            Any: ``_IncompatibleKeys`` named tuple.
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

    @torch.no_grad()
    def preprocess(
        self,
        inputs: dict,
        *,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        range: tuple[float, float] = (0.0, 255.0),
        epsilon: float = 1e-12,
    ) -> dict[str, Any]:
        """
        Normalize planar & spherical images using dataset-provided meta (mean/std/range).
        Performed in pure functional style without any side effects.

        Args:
            inputs (dict): Input dict with keys 'images', 'valid_pixel_masks', 'spherical_images', 'spherical_valid_pixel_masks'.
            mean (tuple[float, float, float] | None, optional): per-channel mean defined on a [0,1] scale. Defaults to None.
            std (tuple[float, float, float] | None, optional): per-channel std defined on a [0,1] scale. Defaults to None.
            range (tuple[float, float], optional): inputs data range before scaling into [0,1]. Defaults to (0.0, 255.0).
            epsilon (float, optional): small number to prevent division by zero NaN. Defaults to 1e-12.

        Returns:
            dict[str, Any]: Normalized inputs.
        """
        planar_images = copy_or_clone(inputs["images"])  # (B, H, W, C) in [0, 255]
        spherical_batch_value = copy_or_clone(
            inputs["spherical_images"].batch_value
        )  # (B, N, C) in [0, 255]
        _device, _dtype = planar_images.device, planar_images.dtype
        planar_valid_pixel_masks = copy_or_clone(
            inputs.get("valid_pixel_masks", torch.ones(planar_images.shape[:-1]))
        ).to(
            dtype=torch.bool, device=_device
        )  # (B, H, W) bool
        if "spherical_valid_pixel_masks" in inputs:
            spherical_batch_value_pixel_masks = copy_or_clone(
                inputs["spherical_valid_pixel_masks"].batch_value
            ).to(
                dtype=torch.bool
            )  # (B, N, 1) bool
        else:
            spherical_batch_value_pixel_masks = torch.ones(
                spherical_batch_value.shape[:-1],
                dtype=torch.bool,
                device=_device,
            ).unsqueeze(
                -1
            )  # (B, N, 1) bool

        # --- validate & prepare constants
        assert (
            planar_images.dim() == 4
        ), f"Expected BHWC, got shape {planar_images.shape}"
        batch_size, height, width, channel_count = planar_images.shape

        if mean is not None:
            mean_tensor = torch.as_tensor(mean, dtype=_dtype, device=_device)
            self.register_buffer("mean", mean_tensor)
        elif getattr(self, "mean", None) is not None:
            mean_tensor = getattr(self, "mean", None)
        else:
            image_net_mean = [0.485, 0.456, 0.406]
            warnings.warn(
                f"Mean not provided by dataset batch metadata, using ImageNet mean {image_net_mean}.",
                UserWarning,
                stacklevel=2,
            )
            # ImageNet mean
            mean_tensor = torch.Tensor(image_net_mean).to(dtype=_dtype, device=_device)
            self.register_buffer("mean", mean_tensor)

        if std is not None:
            std_tensor = torch.as_tensor(std, dtype=_dtype, device=_device)
            self.register_buffer("std", std_tensor)
        elif getattr(self, "std", None) is not None:
            std_tensor = getattr(self, "std", None)
        else:
            image_net_std = [0.229, 0.224, 0.225]
            warnings.warn(
                f"Standard deviation not provided by dataset batch metadata, using ImageNet std {image_net_std}.",
                UserWarning,
                stacklevel=2,
            )
            # ImageNet std
            std_tensor = torch.Tensor(image_net_std).to(dtype=_dtype, device=_device)
            self.register_buffer("std", std_tensor)

        assert (
            mean_tensor.numel() == channel_count
        ), f"mismatch between mean length == {mean_tensor.numel()} and channel count == {channel_count}"
        assert (
            std_tensor.numel() == channel_count
        ), f"mismatch between std length == {std_tensor.numel()} and channel count == {channel_count}"

        input_min = float(range[0])
        input_max = float(range[1])
        scale_denominator = max(input_max - input_min, epsilon)

        # --- planar normalization (BHWC) ---
        planar_images_unit = (planar_images - input_min) / scale_denominator
        normalized_planar_images = (
            planar_images_unit - mean_tensor.view(1, 1, 1, channel_count)
        ) / (std_tensor.view(1, 1, 1, channel_count) + epsilon)
        normalized_planar_images.masked_fill_(
            ~planar_valid_pixel_masks.unsqueeze(-1), 0
        )  # neutral fill invalid pixels

        # --- spherical normalization ---
        spherical_images_unit = (spherical_batch_value - input_min) / scale_denominator
        normalized_spherical_batch_value = (
            spherical_images_unit - mean_tensor.view(1, 1, channel_count)
        ) / (std_tensor.view(1, 1, channel_count) + epsilon)
        normalized_spherical_batch_value.masked_fill_(
            ~spherical_batch_value_pixel_masks, 0
        )

        normalized_batch_spherical_image = inputs["spherical_images"].clone()
        normalized_batch_spherical_image.batch_value = normalized_spherical_batch_value

        return {
            "images": normalized_planar_images,
            "spherical_images": normalized_batch_spherical_image,
        }

    def forward(self, inputs: dict) -> dict:
        """Dispatch to Planar or Spherical backbone and produce class maps.

        Args:
            inputs (dict): Preprocessed inputs with ``"images"`` and ``"spherical_images"``.

        Returns:
            dict: ``"class_maps"`` (B, H, W, C) and ``"spherical_class_maps"`` (B, N, C).
        """
        output_vector_mask = inputs.get("output_vector_mask", None)
        if output_vector_mask is None:
            # if mask is not provided, create a mask of ones
            import warnings

            warnings.warn(
                "Mask not provided for semantic segmentation model, using a mask of ones.",
                UserWarning,
                stacklevel=2,
            )

            output_vector_mask = torch.ones(
                inputs["images"].shape[1:3],
                dtype=torch.bool,
                device=inputs["images"].device,
            )

        if isinstance(self.model, SphericalSemanticSegmentation):
            spherical_maps = self.model(inputs["spherical_images"])
            B, _, C = spherical_maps.batch_value.shape
            H, W = inputs["images"].shape[1:3]
            planar_maps = torch.zeros(
                (B, H, W, C),
                dtype=spherical_maps.batch_value.dtype,
                device=spherical_maps.batch_value.device,
            )
            planar_maps[:, output_vector_mask] = copy_or_clone(
                spherical_maps.batch_value
            ).view(B, -1, C)
        elif isinstance(self.model, PlanarSemanticSegmentation):
            planar_maps = self.model(inputs["images"].permute(0, 3, 1, 2)).permute(
                0, 2, 3, 1
            )  # (B, H, W, C) <=> (B, C, H, W)
            spherical_maps = BatchSphericalImage(
                batch_value=copy_or_clone(planar_maps)[
                    :, output_vector_mask
                ],  # (B, H, W, C) => (B, V, C)
                vector=inputs["spherical_images"].vector,
            )
        else:
            raise TypeError(
                f"Unsupported model type: {type(self.model)}. "
                "Expected SphericalSemanticSegmentation or PlanarSemanticSegmentation."
            )

        return {
            "class_maps": planar_maps,
            "spherical_class_maps": spherical_maps,
        }

    @staticmethod
    def dice_loss(
        predict_maps: BatchSphericalImage,
        gt_maps: BatchSphericalImage,
        class_weight: list[float] | torch.Tensor | None = None,
        ignore_index: int = 0,
        include_absent_classes: bool = False,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        2 x IoU / (predictions + ground truths)
        Multi-class soft Dice loss on flattened valid pixels.
        Computes per-sample Dice, averages across present classes (or all non-ignored if include_absent_classes=True), then averages across batch.

        Args:
            predict_maps (BatchSphericalImage): (B, N, C) logit
            gt_maps (BatchSphericalImage): (B, N, [1 or C])
            class_weight (list[float] | torch.Tensor | None, optional): class weight in cross entropy loss, where minority classes get up-weighted. Defaults to None.
            ignore_index (int, optional): the ignore index. Defaults to 0.
            include_absent_classes (bool, optional): whether to include absent classes in ground truth labels when calculating per-sample dice average. Defaults to False.
            eps (float, optional): small number to prevent division by zero NaN.

        Returns:
            torch.Tensor: dice loss
        """
        # unpack predictions and targets
        logit: torch.Tensor = copy_or_clone(predict_maps.batch_value)  # (B, N, C)
        target_tensor: torch.Tensor = copy_or_clone(
            gt_maps.batch_value
        )  # (B, N, [1 or C])
        batch_size, num_pixels, num_classes = logit.shape

        assert (
            0 <= ignore_index < num_classes
        ), f"ignore_index {ignore_index} out of range, valid range: [0, {num_classes})"

        # convert target to integer IDs
        if target_tensor.shape[-1] == 1:
            target_ids = target_tensor.squeeze(-1).long()  # (B, N)
        else:
            target_ids = target_tensor.argmax(dim=-1).long()

        # ignore pixels with ignore_index
        valid_pixel_mask = target_ids != ignore_index  # (B, N)
        if valid_pixel_mask.sum() == 0:
            return logit.new_tensor(0.0, dtype=torch.float32)
        valid_pixel_mask = valid_pixel_mask.unsqueeze(-1)  # (B, N, 1)

        # convert logits to probabilities
        predicted_probabilities = logit.softmax(dim=-1)  # (B, N, C)

        # apply pixel mask
        predicted_probabilities = (
            predicted_probabilities * valid_pixel_mask
        )  # (B, N, C)

        # build one-hot ground truth tensor
        one_hot_targets = F.one_hot(target_ids.clamp_min(0), num_classes).to(
            predicted_probabilities.dtype
        )
        one_hot_targets = one_hot_targets * valid_pixel_mask  # (B, N, C)

        # compute per-class sums
        predicted_class_sums = predicted_probabilities.sum(dim=-2)  # (B, C)
        target_class_sums = one_hot_targets.sum(dim=-2)  # (B, C)
        intersection_per_class = (predicted_probabilities * one_hot_targets).sum(
            dim=-2
        )  # (B, C)

        # dice per class
        dice_per_class = (2.0 * intersection_per_class + eps) / (
            predicted_class_sums + target_class_sums + eps
        )  # (B, C)

        # which classes count for each sample
        if include_absent_classes:
            present_class_mask = torch.ones_like(
                target_class_sums, dtype=torch.bool
            )  # (B, C)
            present_class_mask[:, ignore_index] = False
        else:
            present_class_mask = target_class_sums > 0  # (B, C)
            present_class_mask[:, ignore_index] = False

        # handle class weights
        if class_weight is not None:
            class_weight_tensor: torch.Tensor = to_torch(
                class_weight, device=logit.device, dtype=torch.float32
            )  # (C,)
            expanded_weights = (
                class_weight_tensor.unsqueeze(0).expand(batch_size, -1)
                * present_class_mask
            )  # (B, C)

            # normalize weights so sum of weights over present classes equals number of present classes
            weight_sums = expanded_weights.sum(dim=1, keepdim=True).clamp_min(
                eps
            )  # (B, C)
            num_present_classes = present_class_mask.sum(dim=1, keepdim=True).clamp_min(
                1.0
            )  # (B, C)
            expanded_weights = (
                expanded_weights * num_present_classes / weight_sums
            )  # (B, C)

            dice_loss_per_sample = ((1.0 - dice_per_class) * expanded_weights).sum(
                dim=1
            ) / num_present_classes.squeeze(
                1
            )  # (B,)
        else:
            num_present_classes = present_class_mask.sum(dim=1).clamp_min(1.0)  # (B,)
            dice_loss_per_sample = ((1.0 - dice_per_class) * present_class_mask).sum(
                dim=1
            ) / num_present_classes

        return dice_loss_per_sample.mean()

    @staticmethod
    def cross_entropy_loss(
        predict_maps: BatchSphericalImage,
        gt_maps: BatchSphericalImage,
        class_weight: list[float] | torch.Tensor | None = None,
        ignore_index: int = 0,
        label_smoothing: float = 0.05,
    ) -> torch.Tensor:
        """
        Cross-entropy over logit with integer class IDs (preferred).
        Also supports legacy one-hot GT by auto-detecting and converting.

        Args:
            predict_maps (BatchSphericalImage): (B, N, C) logit
            gt_maps (BatchSphericalImage): (B, N, [1 or C])
            class_weight (list[float] | torch.Tensor | None, optional): class weight in cross entropy loss, where minority classes get up-weighted. Defaults to None.
            ignore_index (int, optional): the ignore index. Defaults to 0.
            label_smoothing (float, optional): label smoothing factor. Defaults to 0.05.

        Returns:
            torch.Tensor: cross-entropy loss
        """
        assert 0.0 <= label_smoothing <= 1.0, "Label smoothing must be in [0, 1]."

        # flatten valid pixels
        logit: torch.Tensor = copy_or_clone(predict_maps.batch_value)  # (B, N, C)
        target: torch.Tensor = copy_or_clone(gt_maps.batch_value)  # (B, N, [1 or C])

        batch_size, num_pixels, num_classes = logit.shape
        assert (
            0 <= ignore_index < num_classes
        ), f"ignore_index {ignore_index} out of range, valid range: [0, {num_classes})"

        # normalize target to integer IDs (B, N)
        if target.shape[-1] == 1:  # class id (B, N, 1)
            target = target.squeeze(-1)  # (B, N)
        else:  # one-hot (B, N, C)
            target = target.argmax(dim=-1)  # (B, N)

        # class weight
        if class_weight is not None:
            class_weight: torch.Tensor = to_torch(
                class_weight,
                device=target.device,
                dtype=target.dtype,
            )

        return F.cross_entropy(
            logit.transpose(-1, -2),  # CE expects inputs in (B, C, N)
            target.long(),  # CE expects target in (B, N)
            weight=class_weight,
            reduction="mean",
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    @staticmethod
    def loss(
        predict_maps: BatchSphericalImage,
        gt_maps: BatchSphericalImage,
        dice_weight: float = 0.3,
        class_weight: list[float] | torch.Tensor | None = None,
        ignore_index: int = 0,
        label_smoothing: float = 0.05,
        include_absent_classes: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Combo Loss: (1 - dice_weight) x cross_entropy_loss + dice_weight x dice_loss

        Args:
            predict_maps (BatchSphericalImage): predicted (B, N, C)
            gt_maps (BatchSphericalImage): ground truth (B, N, [1 or C])
            dice_weight (float, optional): dice weight
            class_weight (list[float] | torch.Tensor | None, optional): see cross_entropy_loss or dice_loss. Defaults to None.
            ignore_index (int, optional): see cross_entropy_loss or dice_loss. Defaults to 0.
            label_smoothing (float, optional): see cross_entropy_loss. Defaults to 0.05.
            include_absent_classes (bool, optional): see dice_loss. Defaults to False.

        Returns:
            dict[str, torch.Tensor]: Combo Loss and components
        """
        assert (
            0.0 <= dice_weight <= 1.0
        ), f"dice_weight {dice_weight} not in valid range [0.0, 1.0]"

        cross_entropy_loss = SemanticSegmentationLightningModel.cross_entropy_loss(
            predict_maps=predict_maps,
            gt_maps=gt_maps,
            class_weight=class_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        ).contiguous()

        if dice_weight == 0.0:
            return {
                "loss": cross_entropy_loss,
                "cross entropy loss": cross_entropy_loss,
            }

        dice_loss = SemanticSegmentationLightningModel.dice_loss(
            predict_maps=predict_maps,
            gt_maps=gt_maps,
            class_weight=class_weight,
            ignore_index=ignore_index,
            include_absent_classes=include_absent_classes,
        ).contiguous()

        combo_loss = (
            (1 - dice_weight) * cross_entropy_loss + dice_weight * dice_loss
        ).contiguous()

        return {
            "loss": combo_loss,
            "cross entropy loss": cross_entropy_loss,
            "dice loss": dice_loss,
        }

    @staticmethod
    def benchmark(
        predict_maps: BatchSphericalImage,
        gt_maps: BatchSphericalImage,
        ignore_index: int = 0,
    ) -> dict[str, float]:
        """
        Compute mean IoU (mIoU) and mean Accuracy (mAcc) per sample and average across batch.

        Args:
            predict_maps (BatchSphericalImage): logits, shape (B, N, C)
            gt_maps (BatchSphericalImage): one-hot labels or class id, shape (B, N, [1 or C])
            ignore_index (int): index of the class to ignore (e.g., background)

        Returns:
            dict[str, float]:
                - mIoU: mean Intersection over Union
                - mAcc: mean Accuracy
        """
        num_classes = predict_maps.batch_value.shape[-1]
        predict_maps.mask = gt_maps.mask

        # ---- get hard one‑hot preds & gts
        preds = copy_or_clone(predict_maps.batch_value).argmax(dim=-1)  # (B, N)
        preds = F.one_hot(preds, num_classes).bool()  # (B, N, C)
        gts = copy_or_clone(gt_maps.batch_value)  # (B, N, [1 or C])
        if gts.shape[-1] == 1:  # (B, N, 1)
            gts = gts.squeeze(-1)  # (B, N)
            gts = F.one_hot(gts.long(), num_classes).bool()  # (B, N, C)

        # ---- drop ignore_index
        valid_classes = torch.ones(num_classes, device=gts.device, dtype=torch.bool)
        valid_classes[ignore_index] = False
        preds = preds[..., valid_classes]  # (B, N, C')
        gts = gts[..., valid_classes]  # (B, N, C')

        # ---- intersection, union, total_gt
        inter = (preds & gts).sum(dim=1).float()  # (B, C')
        union = (preds | gts).sum(dim=1).float()  # (B, C')
        total = gts.sum(dim=1).float()  # (B, C')

        # ---- clamp denominators so absent‑class gives 0 instead of nan
        iou = inter / union.clamp(min=1)  # safe: 0/1→0
        acc = inter / total.clamp(min=1)  # safe: 0/1→0

        # ---- mask of which classes actually appear
        present = total > 0  # (B, C')

        # ---- per‑sample sum & counts
        iou_sum = (iou * present).sum(dim=1)  # (B,)
        acc_sum = (acc * present).sum(dim=1)  # (B,)
        counts = present.sum(dim=1).clamp(min=1)  # (B,)

        # ---- only average over samples that have *any* valid class
        sample_mask = counts > 0

        mIoU = (
            (iou_sum[sample_mask] / counts[sample_mask]).mean().item()
            if sample_mask.any()
            else 0.0
        )
        mAcc = (
            (acc_sum[sample_mask] / counts[sample_mask]).mean().item()
            if sample_mask.any()
            else 0.0
        )

        return {"mIoU": mIoU, "mAcc": mAcc}

    def on_train_batch_start(
        self,
        batch: dict[str, Any],
        batch_idx: int,
    ) -> int | None:
        """Lightning hook: always proceed (return 0).

        Args:
            batch (dict[str, Any]): Current batch.
            batch_idx (int): Batch index.

        Returns:
            int | None: 0 to continue training.
        """
        return 0  # return -1 will skip the rest of the epoch

    def on_validation_batch_start(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Lightning hook: no-op before each validation batch.

        Args:
            batch (dict[str, Any]): Current batch.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.
        """
        return

    def training_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Preprocess, forward, compute combined loss, optionally benchmark mIoU/mAcc.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Scalar training loss.
        """
        batch_size = batch["inputs"]["images"].shape[0]

        inputs: dict = self.preprocess(batch["inputs"], **batch["meta"]["input"])
        inputs["output_vector_mask"] = batch["meta"]["output_vector_mask"]
        batch["predicts"] = self(inputs)

        # loss
        losses = self.loss(
            predict_maps=batch["predicts"]["spherical_class_maps"],
            gt_maps=batch["labels"]["spherical_class_maps"],
            ignore_index=batch["meta"]["label"]["ignore_index"],
            class_weight=batch["meta"]["label"].get("class_weight", None),
            **self.loss_param,
        )

        self.log_dict(
            {f"train {k}": v for k, v in losses.items()},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            # sync_dist=True,
            batch_size=batch_size,
        )

        with torch.no_grad():
            if self.benchmark_train:
                metrics = self.benchmark(
                    predict_maps=batch["predicts"]["spherical_class_maps"],
                    gt_maps=batch["labels"]["spherical_class_maps"],
                    ignore_index=batch["meta"]["label"]["ignore_index"],
                )
                self.log_dict(
                    {f"train {k}": v for k, v in metrics.items()},
                    prog_bar=True,
                    logger=True,
                    on_epoch=True,
                    # sync_dist=True,
                    batch_size=batch_size,
                )

        return losses["loss"]

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Validate: loss + mIoU/mAcc benchmark + image logging on first batch.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Scalar validation loss.
        """
        dataset = self.trainer.val_dataloaders.dataset
        batch_size = batch["inputs"]["images"].shape[0]

        inputs: dict = self.preprocess(batch["inputs"], **batch["meta"]["input"])
        inputs["output_vector_mask"] = batch["meta"]["output_vector_mask"]
        batch["predicts"] = self(inputs)

        # loss
        losses = self.loss(
            predict_maps=batch["predicts"]["spherical_class_maps"],
            gt_maps=batch["labels"]["spherical_class_maps"],
            ignore_index=batch["meta"]["label"]["ignore_index"],
            class_weight=batch["meta"]["label"].get("class_weight", None),
            **self.loss_param,
        )

        metrics = self.benchmark(
            predict_maps=batch["predicts"]["spherical_class_maps"],
            gt_maps=batch["labels"]["spherical_class_maps"],
            ignore_index=batch["meta"]["label"]["ignore_index"],
        )

        self.log_dict(
            {f"validation {k}": v for k, v in (losses | metrics).items()},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            # sync_dist=True,
            batch_size=batch_size,
        )

        # visualize ground truth and predicts
        if batch_idx == 0 and self.logger is not None:
            gt_figs, _ = dataset.visualize_batch(batch, num_vis=self.num_vis)
            predict_figs, _ = dataset.visualize_batch(
                batch={
                    "inputs": batch["inputs"],
                    "labels": batch["predicts"],
                    "meta": batch["meta"],
                },
                num_vis=self.num_vis,
            )

            gt_images = [fig_to_numpy(fig, dpi=self.log_image_dpi) for fig in gt_figs]
            predict_images = [
                fig_to_numpy(fig, dpi=self.log_image_dpi) for fig in predict_figs
            ]
            for fig in gt_figs + predict_figs:
                plt.close(fig)

            caption = [
                f"Epoch {self.trainer.current_epoch} image {i}"
                for i in range(len(predict_figs))
            ]
            self.log_images(
                images=gt_images,
                captions=caption,
                key="Validation Ground Truth",
            )
            self.log_images(
                images=predict_images,
                captions=caption,
                key="Validation Prediction",
            )

        return losses["loss"]

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Preprocess and store predictions in the batch dict.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            dict: Updated batch with ``"predicts"`` key.
        """
        inputs: dict = self.preprocess(batch["inputs"], **batch["meta"]["input"])
        inputs["output_vector_mask"] = batch["meta"]["output_vector_mask"]
        batch["predicts"] = self(inputs)
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

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure AdamW optimizer with cosine-with-warmup LR scheduler.

        Returns:
            dict[str, Any]: Optimizer and step-level LR scheduler config.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            **self.optimizer_param,
        )
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            **self.lr_scheduler_param,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning rate",
            },
        }
