#!/usr/bin/env python3
#
# Created on Wed Sep 24 2025 16:12:37
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.spherical.aspp import SphericalASPP
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.block.spherical.resnet import SphericalResNet
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.spherical_image import BatchSphericalImage


class SphericalDeepLabV3(nn.Module):
    """Spherical backbone adopting Deep Lab V3 architecture."""

    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        activation: str,
        backend: str,
        weighting_function_config: dict,
        resolution_factor: float,
        location_sampler: str,
        reject_oo_fov_vector: bool,
        block_size: int,
        *args,
        **kwargs,
    ):
        """Initialize the Spherical DeepLabV3 backbone

        Args:
            in_channels (int): Number of channels of the input batch spherical image.
            base_channel (int): Base number of channels used for scaling up in deeper layers.
            activation (str): Activation function type supported by PyTorch.
            backend (str): Backend type for spherical convolution operations.
            weighting_function_config (dict): config for weighting function.
            resolution_factor (float): Ratio of output pixels to input pixels for downsampling (or its reciprocal for upsampling).
                                       This value affects the ring interval in the circle CNN and the radius in circle pooling.
            location_sampler (str): Type of location sampler to use for the output vector of bottom-level spherical layers
            reject_oo_fov_vector (bool): Whether to reject out-of-field-of-view vectors.
            block_size (int): Size of a single block for large pairwise dot operations in circle CNN/pool.
        """
        super().__init__()

        self.backbone = SphericalResNet(
            in_channels=in_channels,
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
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    (
                        "ASPP",
                        SphericalASPP(
                            in_channels=base_channel * 2**5,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            radius_list=[
                                torch.pi * 3 * 1.1 / 100,
                                torch.pi * 3 * 1.1 / 100 * 2,
                                torch.pi * 3 * 1.1 / 100 * 3,
                            ],
                            backend=backend,
                            global_pool=True,
                            weighting_function_config=weighting_function_config,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 3 * 1.1 / 100,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )
        self.feature_upsampler = ValueSampler(
            {
                "value_sampler": "radial_basis_function",
                "value_sampler_config": {
                    "radius": 0.05501648597419262,  # ! temporary hard-code
                    # "num_in_circle_points": 4,
                    "kernel": "gaussian",
                    "sigma": 0.2,
                    "block_size": block_size,
                },
            }
        )
        self.fuser = SphericalCBNA(
            in_channels=base_channel * 2**2,
            out_channels=base_channel * 2**2,
            activation=activation,
            backend=backend,
            weighting_function_config=weighting_function_config,
            radius=torch.pi * 1.2 / 300,
            identical_output_vector=True,
            block_size=block_size,
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: backbone → ASPP classifier → upsample → fuse.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Dense predictions at the input resolution.
        """
        deep_batch_spherical_image = self.backbone(batch_spherical_image)["layer4"]
        upsampled_batch_spherical_image = self.feature_upsampler(
            self.classifier(deep_batch_spherical_image),
            batch_spherical_image.vector,
        )
        out_batch_spherical_image = self.fuser(upsampled_batch_spherical_image)
        return out_batch_spherical_image
