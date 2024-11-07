#!/usr/bin/env python3
#
# Created on Tue Feb 18 2025 19:33:04
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from omegaconf import ListConfig
from torch.nn import Conv2d, Sigmoid

from usf.dataset.iou import pairwiseIoU
from usf.network.block.planar.cbna import PlanarCBNA
from usf.network.block.planar.yolov11 import PlanarYOLOv11
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.block.spherical.yolov11 import SphericalYOLOv11
from usf.network.layer.spherical.activation import SphericalActivation
from usf.network.layer.spherical.conv1x1 import SphericalConv1x1
from usf.sampler.sampler import SphericalSampler
from usf.utils.image import fig_to_numpy
from usf.utils.lr_scheduler import get_cosine_with_min_lr_schedule_with_warmup
from usf.utils.spherical_image import BatchSphericalImage
from usf.utils.torch_numpy import copy_or_clone


class SphericalObjectDetection(nn.Module):
    """Spherical CenterNet-style object detector with four prediction heads (heatmap, center_offset, size, rotation)."""

    def __init__(
        self,
        num_categories: int,
        base_channel: int,
        activation: str,
        backend: str,
        weighting_function_config: dict,
        reject_oo_fov_vector: bool,
        resampler_config: dict[str, Any] | None = None,
        block_size: int = 8192,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the spherical object detection model.

        Args:
            num_categories (int): Number of object categories.
            base_channel (int): Base channel width for the YOLOv11 backbone.
            activation (str): Activation function name.
            backend (str): Spherical conv backend name.
            weighting_function_config (dict): Config for the conv weighting function.
            reject_oo_fov_vector (bool): Whether to mask out-of-FoV vectors.
            resampler_config (dict[str, Any] | None, optional): Optional input resampler config. Defaults to None.
            block_size (int, optional): Block size for pairwise distance computation. Defaults to 8192.
        """
        super().__init__()

        if resampler_config is not None:
            resampler_config["reject_oo_fov_vector"] = reject_oo_fov_vector
            self.input_resampler = SphericalSampler(resampler_config)

        self.backbone = SphericalYOLOv11(
            base_channel=base_channel,
            activation=activation,
            multilevel=False,
            backend=backend,
            weighting_function_config=weighting_function_config,
            reject_oo_fov_vector=reject_oo_fov_vector,
            block_size=block_size,
            *args,
            **kwargs,
        )

        # heatmap head
        self.heatmap_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "linear",
                        SphericalConv1x1(
                            in_channels=base_channel * 2**0,
                            out_channels=num_categories,
                            bias=True,
                        ),
                    ),
                    ("sigmoid", SphericalActivation("Sigmoid")),
                ]
            )
        )

        self.center_offset_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "linear",
                        SphericalConv1x1(
                            in_channels=base_channel * 2**0,
                            out_channels=2,  # S2 coordinate
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        self.size_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "linear",
                        SphericalConv1x1(
                            in_channels=base_channel * 2**0,
                            out_channels=2,  # S2 width, height
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        self.rotation_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "linear",
                        SphericalConv1x1(
                            in_channels=base_channel * 2**0,
                            out_channels=1,  # 1D angle
                            bias=True,
                        ),
                    ),
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> dict[str, BatchSphericalImage]:
        """Forward pass through backbone + 4 detection heads.

        Args:
            batch_spherical_image (BatchSphericalImage): Input spherical image.

        Returns:
            dict[str, BatchSphericalImage]: Per-pixel maps keyed by ``"heatmaps"``,
                ``"center_offsets"``, ``"sizes"``, ``"rotations"``.
        """
        preprocessed_batch_spherical_image = (
            batch_spherical_image
            if getattr(self, "input_resampler", None) is None
            else self.input_resampler(batch_spherical_image)
        )

        feature_batch_spherical_image = self.backbone(
            preprocessed_batch_spherical_image
        )["high_resolution"]

        return {
            "heatmaps": self.heatmap_head(feature_batch_spherical_image),
            "center_offsets": self.center_offset_head(feature_batch_spherical_image),
            "sizes": self.size_head(feature_batch_spherical_image),
            "rotations": self.rotation_head(feature_batch_spherical_image),
        }


class PlanarObjectDetection(nn.Module):
    """Planar object detection model following YOLOv11 architecture."""

    def __init__(
        self,
        num_categories: int,
        base_channel: int,
        activation: str,
        kernel_size: int,
        *args,
        **kwargs,
    ):
        """Initialize PlanarObjectDetection.

        Args:
            num_categories (int): Number of object categories.
            base_channel (int): Base number of channels for feature extraction.
            activation (str): Activation function type.
            kernel_size (int): Kernel size for convolutions (must be odd).
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be an odd number."

        # Backbone
        self.backbone = PlanarYOLOv11(
            base_channel=base_channel,
            activation=activation,
            multilevel=False,
            kernel_size=kernel_size,
            *args,
            **kwargs,
        )

        # Heatmap Head
        self.heatmap_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "linear",
                        Conv2d(
                            in_channels=base_channel * 2**0,
                            out_channels=num_categories,
                            kernel_size=1,
                            stride=1,
                            bias=True,
                        ),
                    ),
                    ("sigmoid", Sigmoid()),
                ]
            )
        )

        # Center Offset Head
        self.center_offset_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "linear",
                        Conv2d(
                            in_channels=base_channel * 2**0,
                            out_channels=2,  # (theta_offset, phi_offset)
                            kernel_size=1,
                            stride=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        # Size Head
        self.size_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "linear",
                        Conv2d(
                            in_channels=base_channel * 2**0,
                            out_channels=2,  # (width, height)
                            kernel_size=1,
                            stride=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

        # Rotation Head
        self.rotation_head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        PlanarCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "cbna2",
                        PlanarCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            activation=activation,
                            kernel_size=kernel_size,
                            stride=1,
                        ),
                    ),
                    (
                        "linear",
                        Conv2d(
                            in_channels=base_channel * 2**0,
                            out_channels=1,  # Angle
                            kernel_size=1,
                            stride=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass of the object detection model."""
        features = self.backbone(x)["high_resolution"]

        return {
            "heatmaps": self.heatmap_head(features),
            "center_offsets": self.center_offset_head(features),
            "sizes": self.size_head(features),
            "rotations": self.rotation_head(features),
        }


class ObjectDetectionLightningModel(pl.LightningModule):
    """Lightning wrapper for CenterNet-style object detection with focal + regression loss and mAP benchmarking."""

    def __init__(
        self,
        optimizer_param: dict,
        lr_scheduler_param: dict,
        benchmark_train: bool,
        benchmark_eval: bool,
        num_regression_points: int,
        extract_raw_rbfovs: dict,
        non_maximum_suppression: dict,
        benchmark: dict,
        num_vis: int,
        log_image_dpi: int,
        loss_param: dict,
        architecture: PlanarObjectDetection | SphericalObjectDetection,
    ) -> None:
        """Initialize the object detection Lightning model.

        Args:
            optimizer_param (dict): Kwargs passed to AdamW.
            lr_scheduler_param (dict): Kwargs passed to cosine LR scheduler.
            benchmark_train (bool): Whether to compute mAP during training.
            benchmark_eval (bool): Whether to compute mAP during validation.
            num_regression_points (int): Number of regression anchor points per detection.
            extract_raw_rbfovs (dict): Config for extracting raw RBFoV detections.
            non_maximum_suppression (dict): Config for NMS post-processing.
            benchmark (dict): Config for mAP benchmark computation.
            num_vis (int): Number of samples to visualize per validation epoch.
            log_image_dpi (int): DPI for logged visualizations.
            loss_param (dict): Parameters for the combined loss function.
            architecture (PlanarObjectDetection | SphericalObjectDetection): The backbone model (Planar or Spherical).
        """
        super().__init__()

        self.optimizer_param = optimizer_param
        self.lr_scheduler_param = lr_scheduler_param

        self.benchmark_train = benchmark_train
        self.benchmark_eval = benchmark_eval
        self.num_regression_points = num_regression_points
        self.extract_raw_rbfovs = extract_raw_rbfovs
        self.non_maximum_suppression = non_maximum_suppression
        self.benchmark_config = benchmark
        self.num_vis = num_vis
        self.log_image_dpi = log_image_dpi
        self.loss_param = loss_param
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
        """Dispatch to Planar or Spherical backbone and produce detection maps.

        Args:
            inputs (dict): Preprocessed inputs with ``"images"`` and ``"spherical_images"``.

        Returns:
            dict: Detection maps (heatmaps, center_offsets, sizes, rotations).
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
        output_vector_mask = copy_or_clone(output_vector_mask)

        if isinstance(self.model, SphericalObjectDetection):
            return self.model(inputs["spherical_images"])
        elif isinstance(self.model, PlanarObjectDetection):
            images = inputs["images"]
            vector = copy_or_clone(inputs["output_vector"])
            image_height, image_width = images.shape[1:3]
            planar_maps = self.model(
                images.permute(0, 3, 1, 2)
            )  # (B, H, W, C) => (B, C, H, W)
            map_height, map_width = planar_maps["heatmaps"].shape[2:4]

            # resolution mismatch, interpolate mask and vector to match the planar maps
            if (map_height, map_width) != (image_height, image_width):
                assert (map_height / map_width) == (
                    image_height / image_width
                ), "Aspect ratio mismatch between input images and predicted planar maps."
                output_vector_mask = (
                    F.interpolate(  # F.interpolate requires input in shape (N, C, H, W)
                        output_vector_mask[None, None].float(),
                        size=(map_height, map_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    > 0.5
                )
                vector = F.normalize(
                    F.interpolate(
                        vector.permute(2, 0, 1).unsqueeze(
                            0
                        ),  # (H, W, 3) => (1, 3, H, W)
                        size=(map_height, map_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0].permute(
                        1, 2, 0
                    ),  # (1, 3, H, W) => (H, W, 3)
                    dim=-1,
                )

            spherical_maps = {}
            for key, value in planar_maps.items():
                batch_value = value.permute(0, 2, 3, 1)  # (B, C, H, W) => (B, H, W, C)
                spherical_maps[key] = BatchSphericalImage(
                    batch_value=batch_value[:, output_vector_mask],
                    vector=vector[output_vector_mask],
                )

            return spherical_maps
        else:
            raise NotImplementedError(
                f"Unsupported model type: {type(self.model)}. "
                "Expected SphericalObjectDetection or PlanarObjectDetection."
            )

    @staticmethod
    def normalize_rotation_differentiable(gamma: torch.Tensor) -> torch.Tensor:
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

    @staticmethod
    def loss(
        predict_maps: dict[str, BatchSphericalImage],
        gt_maps: dict[str, BatchSphericalImage],
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        lambda_size: float = 0.1,
        lambda_offset: float = 0.1,
        lambda_angle: float = 0.1,
        allow_periodic_rotation: bool = False,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor]:
        """
        Loss function for panoramic object detection.

        Args:
            predict_maps (dict[str, BatchSphericalImage]): Predicted maps from model. Expected keys:
                "heatmaps": (B, N, num_categories)
                "center_offsets": (B, N, 2)
                "sizes": (B, N, 2)
                "rotations": (B, N, 1)
                (They all share consistent geometry: vector and polar.)
            gt_maps (dict[str, BatchSphericalImage]): Ground truth maps with keys:
                "heatmaps": (B, N, num_categories)
                "center_offsets": (B, N, 2)
                "sizes": (B, N, 2)
                "rotations": (B, N, 1)
                "masks": (B, N, 1) -- mask for regression supervision.
            focal_alpha (float, optional): Exponent for modulating factor for positive examples. Defaults to 2.0.
            focal_beta (float, optional): Exponent for modulating factor for negative examples. Defaults to 4.0.
            lambda_size (float, optional): Weight for size regression loss. Defaults to 0.1.
            lambda_offset (float, optional): Weight for offset regression loss. Defaults to 0.1.
            lambda_angle (float, optional): Weight for angle regression loss. Defaults to 0.1.
            allow_periodic_rotation (bool, optional): whether to allow predicted angle to be gt angle + k x pi. Defaults to False.
            eps (float, optional): Small constant to avoid division by zero. Defaults to 1e-6.

        Returns:
            dict[str, torch.Tensor]: total loss, classification loss, size loss, offset loss, angle loss
        """
        # Check geometry consistency (only vector).
        assert torch.allclose(
            predict_maps["heatmaps"].vector.to(gt_maps["heatmaps"].vector.dtype),
            gt_maps["heatmaps"].vector,
        ), "inconsistent geometry between predict_maps and gt_maps"

        # Retrieve batch sizes and shapes.
        # heatmap: shape (B, N, num_categories)
        heat_prediction = predict_maps["heatmaps"].batch_value
        heat_gt = gt_maps["heatmaps"].batch_value

        # Modified focal loss for soft Gaussian heatmap.
        # For each pixel, we compute:
        #   loss_pos = - log(p + eps) * (1 - p)^focal_alpha * gt
        #   loss_neg = - log(1 - p + eps) * p^focal_alpha * (1 - gt)^focal_beta
        loss_positive = (
            -torch.log(heat_prediction + eps)
            * torch.pow(1 - heat_prediction, focal_alpha)
            * heat_gt
        )
        loss_negative = (
            -torch.log(1 - heat_prediction + eps)
            * torch.pow(heat_prediction, focal_alpha)
            * torch.pow(1 - heat_gt, focal_beta)
        )
        loss_classification = (loss_positive + loss_negative).sum() / (
            heat_gt.numel() + eps
        )

        # Regression losses (computed only at positive locations indicated by the mask).
        regression_mask = gt_maps["masks"].batch_value  # shape (B, N, 1)

        # Center offset loss (L_off)
        center_offset_prediction = predict_maps[
            "center_offsets"
        ].batch_value  # (B, N, 2)
        center_offset_gt = gt_maps["center_offsets"].batch_value  # (B, N, 2)
        loss_offset = (
            torch.abs(center_offset_prediction - center_offset_gt) * regression_mask
        )
        loss_offset = loss_offset.sum() / (regression_mask.sum() + eps)

        # Size loss (L_size)
        size_prediction = predict_maps["sizes"].batch_value  # (B, N, 2)
        size_gt = gt_maps["sizes"].batch_value  # (B, N, 2)
        loss_size = torch.abs(size_prediction - size_gt) * regression_mask
        loss_size = loss_size.sum() / (regression_mask.sum() + eps)

        # Angle loss (L_direct)
        angle_prediction = predict_maps["rotations"].batch_value  # (B, N, 1)
        if allow_periodic_rotation:  # do not penalize periodicity
            angle_prediction = (
                ObjectDetectionLightningModel.normalize_rotation_differentiable(
                    angle_prediction
                )
            )
        angle_gt = gt_maps["rotations"].batch_value  # (B, N, 1)
        loss_angle = torch.abs(angle_prediction - angle_gt) * regression_mask
        loss_angle = loss_angle.sum() / (regression_mask.sum() + eps)

        # Combine losses
        total_loss = (
            loss_classification
            + lambda_size * loss_size
            + lambda_offset * loss_offset
            + lambda_angle * loss_angle
        ).contiguous()

        return {
            "loss": total_loss,
            "classification loss": loss_classification,
            "size loss": loss_size,
            "offset loss": loss_offset,
            "angle loss": loss_angle,
        }

    @staticmethod
    def single_category_AP(
        predictions_category: torch.Tensor,
        gt_category: torch.Tensor,
        iou_threshold: float | Iterable[float] = 0.5,
        eps: float = 1e-6,
    ) -> float | list[float]:
        """
        Compute Average Precision (AP) for a single category at one or more IoU thresholds,
        vectorizing across thresholds while preserving greedy one-to-one GT matching per threshold.

        Args:
            predictions_category (torch.Tensor): (num_prediction, 6/7), sorted by confidence desc.
            gt_category         (torch.Tensor): (num_gt, 6/7).
            iou_threshold (float | Iterable[float]): single IoU or a list; order preserved in output.
            eps (float): small constant to avoid division by zero.

        Returns:
            float | list[float]: AP(s) in the same order as input thresholds.
        """
        thresholds_is_scalar = isinstance(iou_threshold, (int, float))
        thresholds: torch.Tensor = (
            torch.tensor(
                [float(iou_threshold)],
                device=predictions_category.device,
                dtype=predictions_category.dtype,
            )
            if thresholds_is_scalar
            else torch.tensor(
                [float(t) for t in iou_threshold],
                device=predictions_category.device,
                dtype=predictions_category.dtype,
            )
        )

        if gt_category.numel() == 0:
            return (
                0.0
                if thresholds_is_scalar
                else [0.0 for _ in range(thresholds.numel())]
            )

        num_prediction = predictions_category.shape[0]
        num_gt = gt_category.shape[0]
        num_thresholds = thresholds.numel()

        # IoU matrix once: (P, G) using only geometry cols [:5]
        pairwise_iou = pairwiseIoU(
            predictions_category[:, :5], gt_category[:, :5]
        )  # (P, G)

        # For greedy matching, per prediction take the GT with max IoU (threshold-agnostic)
        best_iou_per_prediction, best_gt_index_per_prediction = pairwise_iou.max(
            dim=1
        )  # (P,), (P,)

        # Per-threshold bookkeeping
        gt_assigned = torch.zeros(
            (num_thresholds, num_gt),
            dtype=torch.bool,
            device=predictions_category.device,
        )
        true_positive_flags = torch.zeros(
            (num_thresholds, num_prediction),
            dtype=predictions_category.dtype,
            device=predictions_category.device,
        )

        # Greedy over predictions (already confidence-sorted)
        for i in range(num_prediction):
            best_iou_i = best_iou_per_prediction[i]  # scalar
            best_gt_idx_i = best_gt_index_per_prediction[i].item()  # int
            if num_gt == 0:
                break

            # Vectorized accept mask over thresholds: (T,)
            accept_mask = (best_iou_i >= thresholds) & (~gt_assigned[:, best_gt_idx_i])

            # Mark TP for thresholds that accept this match
            true_positive_flags[accept_mask, i] = 1.0

            # Assign that GT for those thresholds
            if accept_mask.any():
                gt_assigned[accept_mask, best_gt_idx_i] = True

        # Precision-Recall per threshold
        false_positive_flags = 1.0 - true_positive_flags  # (T, P)
        cumulative_TP = torch.cumsum(true_positive_flags, dim=1)  # (T, P)
        cumulative_FP = torch.cumsum(false_positive_flags, dim=1)  # (T, P)
        precision = cumulative_TP / (cumulative_TP + cumulative_FP + eps)  # (T, P)
        recall = cumulative_TP / (num_gt + eps)  # (T, P)

        # Add boundary points and enforce monotonic precision per threshold
        ones_T = torch.ones(
            (num_thresholds, 1), device=precision.device, dtype=precision.dtype
        )
        zeros_T = torch.zeros(
            (num_thresholds, 1), device=precision.device, dtype=precision.dtype
        )

        precision = torch.cat([ones_T, precision, zeros_T], dim=1)  # (T, P+2)
        recall = torch.cat([zeros_T, recall, ones_T], dim=1)  # (T, P+2)

        # Monotone precision from right to left
        # (vectorized cumulative maximum along prediction axis)
        for k in range(precision.shape[1] - 2, -1, -1):
            precision[:, k] = torch.maximum(precision[:, k], precision[:, k + 1])

        # Area under PR for each threshold (trapz along prediction axis)
        ap_per_threshold = torch.trapz(precision, recall, dim=1)  # (T,)
        ap_list = ap_per_threshold.tolist()
        return ap_list[0] if thresholds_is_scalar else ap_list

    @staticmethod
    def benchmark(
        predict_rbfovs: list[torch.Tensor],
        gt_rbfovs: list[torch.Tensor],
        iou_threshold: float | Iterable[float] = 0.5,
    ) -> float | list[float]:
        """
        Compute batch-level mAP at one or more IoU thresholds by averaging per-image AP,
        where each image AP is the mean AP across all GT categories present in that image.

        Args:
            predict_rbfovs (list[torch.Tensor]):
                list of predicted RBFoVs per image.
                Each tensor has shape (num_prediction, 7) with format
                [θ, φ, α, β, γ, category, confidence].
            gt_rbfovs (list[torch.Tensor]):
                list of ground-truth RBFoVs per image.
                Each tensor has shape (num_gt, 7) with the same format (no confidence field used).
            iou_threshold (float | Iterable[float], optional):
                IoU threshold(s). If a single float is provided, a single float mAP is returned.
                If a sequence is provided, a list of mAPs (same order) is returned.

        Returns:
            float | list[float]:
                mAP at the specified IoU threshold(s). Order matches the input thresholds.

        Notes:
            - Only categories that appear in GT for a given image contribute to that image's AP.
            - If an image has no GT at all, it is skipped in the averaging.
        """
        thresholds_is_scalar = isinstance(iou_threshold, (int, float))
        thresholds: list[float] = (
            [float(iou_threshold)] if thresholds_is_scalar else [float(t) for t in iou_threshold]  # type: ignore[arg-type]
        )
        for t in thresholds:
            assert 0.0 < t < 1.0, f"Expect iou_threshold in (0, 1), got {t}"

        # For multiple thresholds, maintain parallel accumulators (one per threshold).
        image_AP_per_threshold: list[list[float]] = [[] for _ in thresholds]

        for predict_sample, gt_sample in zip(predict_rbfovs, gt_rbfovs):
            if gt_sample.numel() == 0:
                continue  # no ground-truth in this image → skip entirely

            categories_in_image = gt_sample[:, 5].unique()
            if categories_in_image.numel() == 0:
                continue

            # For each threshold, collect this image's AP (mean over its GT categories).
            # We compute per-category AP per threshold, then average across categories.
            per_threshold_category_APs: list[list[float]] = [[] for _ in thresholds]

            for category_id in categories_in_image:
                predictions_category = predict_sample[
                    predict_sample[:, 5] == category_id
                ]
                gt_category = gt_sample[gt_sample[:, 5] == category_id]

                if predictions_category.numel() == 0:
                    # No predictions for this category → AP = 0 for all thresholds
                    for k in range(len(thresholds)):
                        per_threshold_category_APs[k].append(0.0)
                    continue

                # Compute AP(s) once; returns float or list depending on thresholds length.
                ap_values = ObjectDetectionLightningModel.single_category_AP(
                    predictions_category=predictions_category,
                    gt_category=gt_category,
                    iou_threshold=thresholds,  # always pass list here
                )
                # ap_values is list[float] because we passed a list of thresholds
                for k, ap_k in enumerate(ap_values):  # type: ignore[index]
                    per_threshold_category_APs[k].append(ap_k)

            # Aggregate over categories for this image, per threshold.
            for k in range(len(thresholds)):
                if len(per_threshold_category_APs[k]) > 0:
                    image_AP_per_threshold[k].append(
                        sum(per_threshold_category_APs[k])
                        / len(per_threshold_category_APs[k])
                    )

        # Final aggregation over images, per threshold.
        if all(len(v) == 0 for v in image_AP_per_threshold):
            return 0.0 if thresholds_is_scalar else [0.0 for _ in thresholds]

        mAPs = [
            (sum(vals) / len(vals)) if len(vals) > 0 else 0.0
            for vals in image_AP_per_threshold
        ]
        return mAPs[0] if thresholds_is_scalar else mAPs

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
        """Preprocess, forward, compute focal + regression loss, optionally benchmark mAP.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            torch.Tensor: Scalar training loss.
        """
        dataset = self.trainer.train_dataloader.dataset
        batch_size = batch["inputs"]["images"].shape[0]

        inputs: dict = self.preprocess(batch["inputs"], **batch["meta"]["input"])
        inputs["output_vector_mask"] = batch["meta"]["output_vector_mask"]
        inputs["output_vector"] = batch["meta"]["output_vector"]
        batch["predicts"] = {}
        batch["predicts"]["maps"] = self(inputs)

        # get ground truth maps
        batch["labels"]["maps"] = dataset.generate_gt_maps(
            rbfovs_list=batch["labels"]["converted_rbfovs"],
            vector=batch["predicts"]["maps"]["heatmaps"].vector,
            n_points=self.num_regression_points,
        )

        # loss
        losses = self.loss(
            predict_maps=batch["predicts"]["maps"],
            gt_maps=batch["labels"]["maps"],
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
                # get predicted rbfov
                batch["predicts"]["converted_rbfovs"] = dataset.extract_raw_rbfovs(
                    batch["predicts"]["maps"],
                    **self.extract_raw_rbfovs,
                )
                batch["predicts"]["converted_rbfovs"] = dataset.non_maximum_suppression(
                    batch["predicts"]["converted_rbfovs"],
                    **self.non_maximum_suppression,
                )
                mAP = self.benchmark(
                    predict_rbfovs=batch["predicts"]["converted_rbfovs"],
                    gt_rbfovs=batch["labels"]["converted_rbfovs"],
                    **self.benchmark_config,
                )
                if not isinstance(mAP, list):
                    mAP = [mAP]
                mAP_iou_threshold = self.benchmark_config["iou_threshold"]
                if isinstance(mAP_iou_threshold, ListConfig):
                    mAP_iou_threshold = list(mAP_iou_threshold)
                elif isinstance(mAP_iou_threshold, float):
                    mAP_iou_threshold = [float(mAP_iou_threshold)]
                self.log_dict(
                    {
                        f"train mAP@{int(iou_threshold * 100)}": value
                        for iou_threshold, value in zip(mAP_iou_threshold, mAP)
                    },
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
        """Validate: loss + optional mAP benchmark + image logging.

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
        inputs["output_vector"] = batch["meta"]["output_vector"]
        batch["predicts"] = {}
        batch["predicts"]["maps"] = self(inputs)

        # get ground truth maps
        batch["labels"]["maps"] = dataset.generate_gt_maps(
            rbfovs_list=batch["labels"]["converted_rbfovs"],
            vector=batch["predicts"]["maps"]["heatmaps"].vector,
            n_points=self.num_regression_points,
        )

        # loss
        losses = self.loss(
            predict_maps=batch["predicts"]["maps"],
            gt_maps=batch["labels"]["maps"],
            **self.loss_param,
        )

        self.log_dict(
            {f"validation {k}": v for k, v in losses.items()},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            # sync_dist=True,
            batch_size=batch_size,
        )

        if self.benchmark_eval:
            # get predicted rbfov
            batch["predicts"]["converted_rbfovs"] = dataset.extract_raw_rbfovs(
                batch["predicts"]["maps"],
                **self.extract_raw_rbfovs,
            )
            batch["predicts"]["converted_rbfovs"] = dataset.non_maximum_suppression(
                batch["predicts"]["converted_rbfovs"],
                **self.non_maximum_suppression,
            )
            mAP = self.benchmark(
                predict_rbfovs=batch["predicts"]["converted_rbfovs"],
                gt_rbfovs=batch["labels"]["converted_rbfovs"],
                **self.benchmark_config,
            )
            if not isinstance(mAP, list):
                mAP = [mAP]
            mAP_iou_threshold = self.benchmark_config["iou_threshold"]
            if isinstance(mAP_iou_threshold, ListConfig):
                mAP_iou_threshold = list(mAP_iou_threshold)
            elif isinstance(mAP_iou_threshold, float):
                mAP_iou_threshold = [mAP_iou_threshold]

            self.log_dict(
                {
                    f"validation mAP@{int(iou_threshold * 100)}": value
                    for iou_threshold, value in zip(mAP_iou_threshold, mAP)
                },
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

                gt_images = [
                    fig_to_numpy(fig, dpi=self.log_image_dpi) for fig in gt_figs
                ]
                predict_images = [
                    fig_to_numpy(fig, dpi=self.log_image_dpi) for fig in predict_figs
                ]
                for fig in gt_figs + predict_figs:
                    plt.close(fig)

                caption = [
                    f"Epoch {self.trainer.current_epoch} image {i}"
                    for i in range(len(gt_figs))
                ]
                self.log_images(gt_images, caption, key="Validation Ground Truth")
                self.log_images(predict_images, caption, key="Validation Prediction")

        return losses["loss"]

    def predict_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        """Preprocess and store detection map predictions in the batch dict.

        Args:
            batch (dict): Batch from dataloader.
            batch_idx (int): Batch index.
            dataloader_idx (int, optional): Dataloader index. Defaults to 0.

        Returns:
            dict: Updated batch with ``"predicts"`` key.
        """
        inputs: dict = self.preprocess(batch["inputs"], **batch["meta"]["input"])
        inputs["output_vector_mask"] = batch["meta"]["output_vector_mask"]
        inputs["output_vector"] = batch["meta"]["output_vector"]
        batch["predicts"] = {}
        batch["predicts"]["maps"] = self(inputs)
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
