#!/usr/bin/env python3
#
# Created on Tue Feb 25 2025 16:00:47
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.spherical.c2psa import SphericalC2PSA
from usf.network.block.spherical.c3k2 import SphericalC3K2
from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.block.spherical.sppf import SphericalSPPF
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalYOLOv11(nn.Module):
    """Spherical task-agnostic backbone adopting YOLO v11 architecture"""

    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        activation: str,
        expansion: float,
        backend: str,
        weighting_function_config: dict,
        multilevel: bool,
        num_psa: int,
        resolution_factor: float,
        location_sampler: str,
        reject_oo_fov_vector: bool,
        block_size: int,
        attention_backend: str,
        *args,
        **kwargs,
    ):
        """Initialize the Spherical YOLOv11 backbone

        Args:
            in_channels (int): Number of channels of the input batch spherical image.
            base_channel (int): Base number of channels used for scaling up in deeper layers.
            activation (str): Activation function type supported by PyTorch.
            expansion (float): Expansion ratio for the number of channels in hidden layers.
            backend (str): Backend type for spherical convolution operations.
            weighting_function_config (dict): Config for weighting function.
            multilevel (bool): Whether to output multi-level features from different stages.
            num_psa (int): Number of PSA (Polarized Self-Attention) blocks in the neck.
            resolution_factor (float): Ratio of output pixels to input pixels for downsampling.
            location_sampler (str): Type of location sampler for the output vector.
            reject_oo_fov_vector (bool): Whether to reject out-of-field-of-view vectors.
            block_size (int): Block size for pairwise dot operations.
            attention_backend (str): Backend for self-attention layers.
        """
        super().__init__()

        assert (
            resolution_factor > 0.0
        ), f"backbone resolution_factor must be positive, got {resolution_factor}"

        self.downsample1 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=base_channel * 2**0,
            backend=backend,
            weighting_function_config=weighting_function_config,
            radius=torch.pi * 6 / 800,
            activation=activation,
            location_sampler=location_sampler,
            reject_oo_fov_vector=reject_oo_fov_vector,
            resolution_factor=resolution_factor,
            block_size=block_size,
        )

        self.downsample2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**1,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 400,
                            activation=activation,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            resolution_factor=resolution_factor,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "c3k2",
                        SphericalC3K2(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            activation=activation,
                            expansion=expansion,
                            num_c3k=3,
                            shortcut=False,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 192,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )

        self.downsample3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**2,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 240,
                            activation=activation,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            resolution_factor=resolution_factor,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "c3k2",
                        SphericalC3K2(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            activation=activation,
                            expansion=expansion,
                            num_c3k=6,
                            shortcut=False,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 90,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )

        self.downsample4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**3,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.3 / 120,
                            activation=activation,
                            backend=backend,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            resolution_factor=resolution_factor,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "c3k2",
                        SphericalC3K2(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            activation=activation,
                            expansion=expansion,
                            num_c3k=6,
                            shortcut=True,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 45,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )

        self.downsample5 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**4,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.15 / 60,
                            activation=activation,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            resolution_factor=resolution_factor,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "c3k2",
                        SphericalC3K2(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            expansion=expansion,
                            num_c3k=3,
                            shortcut=True,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 2 / 45,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )

        self.neck = nn.Sequential(
            OrderedDict(
                [
                    (
                        "sppf",
                        SphericalSPPF(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            activation=activation,
                            num_pooling_layers=3,  # from YOLOv11
                            pool_type="max",
                            radius=torch.pi / 13,
                            expansion=expansion,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "c2psa",
                        SphericalC2PSA(
                            in_channels=base_channel * 2**4,
                            activation=activation,
                            num_psa=num_psa,
                            shortcut=True,
                            expansion=expansion,
                            backend=attention_backend,
                        ),
                    ),
                ]
            )
        )

        self.upsample1 = SphericalCBNA(
            in_channels=base_channel * 2**4,
            out_channels=base_channel * 2**3,
            # backend=backend,
            # weighting_function_config=weighting_function_config,
            # radius=torch.pi * 6 / 24,
            activation=activation,
            # location_sampler=location_sampler,
            # reject_oo_fov_vector=reject_oo_fov_vector,
            # resolution_factor=1 / resolution_factor,
            # block_size=block_size,
            interpolation={
                "value_sampler": "radial_basis_function",
                "value_sampler_config": {
                    "radius": 0.225,
                    # "num_in_circle_points": 4,
                    "kernel": "gaussian",
                    "sigma": 0.2,
                    "block_size": block_size,
                },
            },
        )

        self.c3k2_upsample1 = SphericalC3K2(
            in_channels=base_channel * 2**4,
            out_channels=base_channel * 2**3,
            activation=activation,
            expansion=expansion,
            num_c3k=3,
            shortcut=True,
            backend=backend,
            weighting_function_config=weighting_function_config,
            radius=torch.pi * 6 / 45,
            block_size=block_size,
        )

        self.upsample2 = SphericalCBNA(
            in_channels=base_channel * 2**3,
            out_channels=base_channel * 2**2,
            # backend=backend,
            # weighting_function_config=weighting_function_config,
            # radius=torch.pi * 6 / 60 * 1.25,
            activation=activation,
            # location_sampler=location_sampler,
            # reject_oo_fov_vector=reject_oo_fov_vector,
            # resolution_factor=1 / resolution_factor,
            # block_size=block_size,
            interpolation={
                "value_sampler": "radial_basis_function",
                "value_sampler_config": {
                    "radius": 0.12762142345309258,
                    # "num_in_circle_points": 4,
                    "kernel": "gaussian",
                    "sigma": 0.2,
                    "block_size": block_size,
                },
            },
        )

        self.c3k2_upsample2 = SphericalC3K2(
            in_channels=base_channel * 2**3,
            out_channels=base_channel * 2**2,
            activation=activation,
            expansion=expansion,
            num_c3k=3,
            shortcut=True,
            backend=backend,
            weighting_function_config=weighting_function_config,
            radius=torch.pi * 6 / 90,
            block_size=block_size,
        )

        self.multilevel = multilevel
        if self.multilevel:
            # Define more downsample
            self.downsample6 = SphericalCBNA(
                in_channels=base_channel * 2**2,
                out_channels=base_channel * 2**3,
                backend=backend,
                weighting_function_config=weighting_function_config,
                radius=torch.pi * 6 * 1.3 / 120,
                activation=activation,
                location_sampler=location_sampler,
                reject_oo_fov_vector=reject_oo_fov_vector,
                resolution_factor=resolution_factor,
                block_size=block_size,
            )
            self.c3k2_downsample6 = SphericalC3K2(
                in_channels=base_channel * 2**4,
                out_channels=base_channel * 2**3,
                activation=activation,
                expansion=expansion,
                num_c3k=3,
                shortcut=False,
                backend=backend,
                weighting_function_config=weighting_function_config,
                radius=torch.pi * 6 / 45,
                block_size=block_size,
            )

            self.downsample7 = SphericalCBNA(
                in_channels=base_channel * 2**3,
                out_channels=base_channel * 2**4,
                backend=backend,
                weighting_function_config=weighting_function_config,
                radius=torch.pi * 6 * 1.15 / 60,
                activation=activation,
                location_sampler=location_sampler,
                resolution_factor=resolution_factor,
                block_size=block_size,
            )
            self.c3k2_downsample7 = SphericalC3K2(
                in_channels=base_channel * 2**5,
                out_channels=base_channel * 2**4,
                activation=activation,
                expansion=expansion,
                num_c3k=3,
                shortcut=False,
                backend=backend,
                weighting_function_config=weighting_function_config,
                radius=torch.pi * 6 * 2 / 45,
                block_size=block_size,
            )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> dict[str, BatchSphericalImage]:
        """Forward pass through the spherical YOLOv11 backbone+neck.

        Down-samples in 5 stages, applies the neck, then up-samples with
        feature fusion. Returns multi-scale feature maps keyed by stage name.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            dict[str, BatchSphericalImage]: Multi-scale feature maps.
        """
        # Downsample
        down1_batch_spherical_image = self.downsample1(batch_spherical_image)
        down2_batch_spherical_image = self.downsample2(down1_batch_spherical_image)
        down3_batch_spherical_image = self.downsample3(down2_batch_spherical_image)
        down4_batch_spherical_image = self.downsample4(down3_batch_spherical_image)
        down5_batch_spherical_image = self.downsample5(down4_batch_spherical_image)

        # Neck
        deep_batch_spherical_image = self.neck(down5_batch_spherical_image)

        # Upsample
        self.upsample1[0].set_output_vector(down4_batch_spherical_image.vector)
        up1_batch_spherical_image = self.c3k2_upsample1(
            concatenate(
                [
                    down4_batch_spherical_image,
                    self.upsample1(deep_batch_spherical_image),
                ]
            )
        )
        self.upsample2[0].set_output_vector(down3_batch_spherical_image.vector)
        up2_batch_spherical_image = self.c3k2_upsample2(
            concatenate(
                [
                    down3_batch_spherical_image,
                    self.upsample2(up1_batch_spherical_image),
                ]
            )
        )

        output_dict = {
            "high_resolution": up2_batch_spherical_image,
            # downstream layers might need first few layers for concatenation
            "down1": down1_batch_spherical_image,
            "down2": down2_batch_spherical_image,
        }

        if self.multilevel:
            # Downsample
            self.downsample6[0].set_output_vector(up1_batch_spherical_image.vector)
            down6_batch_spherical_image = self.c3k2_downsample6(
                concatenate(
                    [
                        up1_batch_spherical_image,
                        self.downsample6(up2_batch_spherical_image),
                    ]
                )
            )
            output_dict["mid_resolution"] = down6_batch_spherical_image

            self.downsample7[0].set_output_vector(deep_batch_spherical_image.vector)
            down7_batch_spherical_image = self.c3k2_downsample7(
                concatenate(
                    [
                        deep_batch_spherical_image,
                        self.downsample7(down6_batch_spherical_image),
                    ]
                )
            )
            output_dict["low_resolution"] = down7_batch_spherical_image

        return output_dict
