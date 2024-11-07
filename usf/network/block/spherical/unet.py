#!/usr/bin/env python3
#
# Created on Tue Aug 26 2025 17:16:12
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch
import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.layer.spherical.circle_pool import CirclePool
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalUNet(nn.Module):
    """Spherical backbone adopting UNet architecture"""

    def __init__(
        self,
        in_channels: int,
        base_channel: int,
        activation: str,
        location_sampler: str,
        backend: str,
        weighting_function_config: dict,
        resolution_factor: float,
        reject_oo_fov_vector: bool,
        block_size: int,
        *args,
        **kwargs,
    ):
        """Initialize the Spherical UNet backbone

        Args:
            in_channels (int): Number of channels of the input batch spherical image.
            base_channel (int): Base number of channels used for scaling up in deeper layers.
            activation (str): Activation function type supported by PyTorch.
            location_sampler (str): Type of location sampler for the output vector.
            backend (str): Spherical conv backend name.
            weighting_function_config (dict): Config for weighting function.
            resolution_factor (float): Ratio of output pixels to input pixels for downsampling.
                                       This value affects the ring interval in the circle CNN and the radius in circle pooling.
            reject_oo_fov_vector (bool): Whether to reject out-of-field-of-view vectors.
            block_size (int): Size of a single block for large pairwise dot operations in circle CNN/pool.
        """
        super().__init__()

        assert (
            resolution_factor > 0.0
        ), f"backbone resolution_factor must be positive, got {resolution_factor}"

        self.downsample1 = nn.Sequential(
            OrderedDict(
                [
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=in_channels,
                    #         out_channels=base_channel * 2**0,
                    #         radius=torch.pi * 6 / 800,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**0,
                    #         out_channels=base_channel * 2**0,
                    #         radius=torch.pi * 6 / 800,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=in_channels,
                            out_channels=base_channel * 2**0,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 800,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**0,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 800,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=in_channels,
                    #         out_channels=base_channel * 2**0,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=3,
                    #         shortcut=False,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.1 / 1600,
                    #         block_size=block_size,
                    #     ),
                    # ),
                ]
            )
        )

        self.downsample2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 3 * 1.2 / 1600,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**0,
                    #         out_channels=base_channel * 2**1,
                    #         radius=torch.pi * 6 / 400,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**1,
                    #         radius=torch.pi * 6 / 400,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**1,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 400,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 400,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**0,
                    #         out_channels=base_channel * 2**1,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=3,
                    #         shortcut=False,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.15 / 800,
                    #         block_size=block_size,
                    #     ),
                    # ),
                ]
            )
        )

        self.downsample3 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 3 * 1.15 / 800,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**2,
                    #         radius=torch.pi * 6 / 240,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**2,
                    #         radius=torch.pi * 6 / 240,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**2,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 240,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 240,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**2,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=6,
                    #         shortcut=True,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.3 / 480,
                    #         block_size=block_size,
                    #     ),
                    # ),
                ]
            )
        )

        self.downsample4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 3 * 1.3 / 480,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**3,
                    #         radius=torch.pi * 6 * 1.3 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**3,
                    #         radius=torch.pi * 6 * 1.3 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**3,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.3 / 120,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.3 / 120,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**3,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=6,
                    #         shortcut=True,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.3 / 240,
                    #         block_size=block_size,
                    #     ),
                    # ),
                ]
            )
        )

        self.neck = nn.Sequential(
            OrderedDict(
                [
                    (
                        "pool",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 3 * 1.5 / 240,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**4,
                    #         radius=torch.pi * 6 * 1.15 / 60,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**4,
                    #         out_channels=base_channel * 2**4,
                    #         radius=torch.pi * 6 * 1.15 / 60,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**4,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.15 / 60,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**4,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 * 1.15 / 60,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**4,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=6,
                    #         shortcut=True,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.15 / 120,
                    #         block_size=block_size,
                    #     ),
                    # ),
                    # (
                    #     "upconv",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**4,
                    #         out_channels=base_channel * 2**3,
                    #         radius=torch.pi * 6 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         resolution_factor=1 / resolution_factor,
                    #         reject_oo_fov_vector=reject_oo_fov_vector,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act3", SphericalActivation(activation=activation)),
                    (
                        "upconv",
                        SphericalCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**3,
                            # backend=backend,
                            # weighting_function_config=weighting_function_config,
                            # radius=torch.pi * 6 * 1.5 / 240,
                            activation=activation,
                            # location_sampler=location_sampler,
                            # reject_oo_fov_vector=reject_oo_fov_vector,
                            # resolution_factor=1 / resolution_factor,
                            # block_size=block_size,
                            interpolation={
                                "value_sampler": "radial_basis_function",
                                "value_sampler_config": {
                                    "radius": 0.127856083214283,
                                    # "num_in_circle_points": 4,
                                    "kernel": "gaussian",
                                    "sigma": 0.2,
                                    "block_size": block_size,
                                },
                            },
                        ),
                    ),
                ]
            )
        )

        self.upsample1 = nn.Sequential(
            OrderedDict(
                [
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**4,
                    #         out_channels=base_channel * 2**3,
                    #         radius=torch.pi * 6 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**3,
                    #         radius=torch.pi * 6 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**4,
                            out_channels=base_channel * 2**3,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 120,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**3,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 120,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**4,
                    #         out_channels=base_channel * 2**3,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=6,
                    #         shortcut=True,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.4 / 240,
                    #         block_size=block_size,
                    #     ),
                    # ),
                    # (
                    #     "upconv",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**2,
                    #         radius=torch.pi * 6 / 120,
                    #         weighting_function_config=weighting_function_config,
                    #         resolution_factor=1 / resolution_factor,
                    #         reject_oo_fov_vector=reject_oo_fov_vector,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act3", SphericalActivation(activation=activation)),
                    (
                        "upconv",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**2,
                            # backend=backend,
                            # weighting_function_config=weighting_function_config,
                            # radius=torch.pi * 6 * 1.4 / 240,
                            activation=activation,
                            # location_sampler=location_sampler,
                            # reject_oo_fov_vector=reject_oo_fov_vector,
                            # resolution_factor=1 / resolution_factor,
                            # block_size=block_size,
                            interpolation={
                                "value_sampler": "radial_basis_function",
                                "value_sampler_config": {
                                    "radius": 0.05502985045313835,
                                    # "num_in_circle_points": 4,
                                    "kernel": "gaussian",
                                    "sigma": 0.2,
                                    "block_size": block_size,
                                },
                            },
                        ),
                    ),
                ]
            )
        )

        self.upsample2 = nn.Sequential(
            OrderedDict(
                [
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**2,
                    #         radius=torch.pi * 6 / 200,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**2,
                    #         radius=torch.pi * 6 / 200,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**3,
                            out_channels=base_channel * 2**2,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 200,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**2,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 200,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**3,
                    #         out_channels=base_channel * 2**2,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=6,
                    #         shortcut=True,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.1 / 400,
                    #         block_size=block_size,
                    #     ),
                    # ),
                    # (
                    #     "upconv",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**1,
                    #         radius=torch.pi * 6 / 400,
                    #         weighting_function_config=weighting_function_config,
                    #         resolution_factor=1 / resolution_factor,
                    #         reject_oo_fov_vector=reject_oo_fov_vector,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act3", SphericalActivation(activation=activation)),
                    (
                        "upconv",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            # backend=backend,
                            # weighting_function_config=weighting_function_config,
                            # radius=torch.pi * 6 * 1.1 / 400,
                            activation=activation,
                            # location_sampler=location_sampler,
                            # reject_oo_fov_vector=reject_oo_fov_vector,
                            # resolution_factor=1 / resolution_factor,
                            # block_size=block_size,
                            interpolation={
                                "value_sampler": "radial_basis_function",
                                "value_sampler_config": {
                                    "radius": 0.02489995490759611,
                                    # "num_in_circle_points": 4,
                                    "kernel": "gaussian",
                                    "sigma": 0.2,
                                    "block_size": block_size,
                                },
                            },
                        ),
                    ),
                ]
            )
        )

        self.upsample3 = nn.Sequential(
            OrderedDict(
                [
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**1,
                    #         radius=torch.pi * 6 / 400,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**1,
                    #         radius=torch.pi * 6 / 400,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**2,
                            out_channels=base_channel * 2**1,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 400,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**1,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 400,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**2,
                    #         out_channels=base_channel * 2**1,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=3,
                    #         shortcut=False,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.1 / 800,
                    #         block_size=block_size,
                    #     ),
                    # ),
                    # (
                    #     "upconv",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**0,
                    #         radius=torch.pi * 6 / 800,
                    #         weighting_function_config=weighting_function_config,
                    #         resolution_factor=1 / resolution_factor,
                    #         reject_oo_fov_vector=reject_oo_fov_vector,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act3", SphericalActivation(activation=activation)),
                    (
                        "upconv",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            # backend=backend,
                            # weighting_function_config=weighting_function_config,
                            # radius=torch.pi * 6 * 1.1 / 800,
                            activation=activation,
                            # location_sampler=location_sampler,
                            # reject_oo_fov_vector=reject_oo_fov_vector,
                            # resolution_factor=1 / resolution_factor,
                            # block_size=block_size,
                            interpolation={
                                "value_sampler": "radial_basis_function",
                                "value_sampler_config": {
                                    "radius": 0.011838254984468222,
                                    # "num_in_circle_points": 4,
                                    "kernel": "gaussian",
                                    "sigma": 0.2,
                                    "block_size": block_size,
                                },
                            },
                        ),
                    ),
                ]
            )
        )

        self.upsample4 = nn.Sequential(
            OrderedDict(
                [
                    # (
                    #     "conv1",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**0,
                    #         radius=torch.pi * 6 / 800,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act1", SphericalActivation(activation=activation)),
                    # (
                    #     "conv2",
                    #     GenericSphericalConv(
                    #         backend=backend,
                    #         in_channels=base_channel * 2**0,
                    #         out_channels=base_channel * 2**0,
                    #         radius=torch.pi * 6 / 800,
                    #         weighting_function_config=weighting_function_config,
                    #         identical_output_vector=True,
                    #         block_size=block_size,
                    #         bias=True,
                    #     ),
                    # ),
                    # ("act2", SphericalActivation(activation=activation)),
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=base_channel * 2**1,
                            out_channels=base_channel * 2**0,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 800,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=base_channel * 2**0,
                            out_channels=base_channel * 2**0,
                            backend=backend,
                            weighting_function_config=weighting_function_config,
                            radius=torch.pi * 6 / 800,
                            activation=activation,
                            identical_output_vector=True,
                            block_size=block_size,
                        ),
                    ),
                    # (
                    #     "c3k2",
                    #     SphericalC3K2(
                    #         in_channels=base_channel * 2**1,
                    #         out_channels=base_channel * 2**0,
                    #         activation=activation,
                    #         expansion=expansion,
                    #         num_c3k=3,
                    #         shortcut=False,
                    #         backend=backend,
                    #         weighting_function_config=weighting_function_config,
                    #         radius=torch.pi * 6 * 1.05 / 1600,
                    #         block_size=block_size,
                    #     ),
                    # ),
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass through the spherical UNet encoder–decoder.

        Down-samples through 4 stages, applies the neck, then up-samples with
        skip connections back to the input resolution.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Dense output at the original resolution.
        """
        # Downsample
        down1_batch_spherical_image = self.downsample1(batch_spherical_image)
        down2_batch_spherical_image = self.downsample2(down1_batch_spherical_image)
        down3_batch_spherical_image = self.downsample3(down2_batch_spherical_image)
        down4_batch_spherical_image = self.downsample4(down3_batch_spherical_image)

        # Neck
        self.neck.upconv[0].set_output_vector(down4_batch_spherical_image.vector)
        deep_batch_spherical_image = self.neck(down4_batch_spherical_image)

        # Upsample
        self.upsample1.upconv[0].set_output_vector(down3_batch_spherical_image.vector)
        up1_batch_spherical_image = self.upsample1(
            concatenate([down4_batch_spherical_image, deep_batch_spherical_image])
        )
        self.upsample2.upconv[0].set_output_vector(down2_batch_spherical_image.vector)
        up2_batch_spherical_image = self.upsample2(
            concatenate([down3_batch_spherical_image, up1_batch_spherical_image])
        )
        self.upsample3.upconv[0].set_output_vector(down1_batch_spherical_image.vector)
        up3_batch_spherical_image = self.upsample3(
            concatenate([down2_batch_spherical_image, up2_batch_spherical_image])
        )
        up4_batch_spherical_image = self.upsample4(
            concatenate([down1_batch_spherical_image, up3_batch_spherical_image])
        )

        return up4_batch_spherical_image
