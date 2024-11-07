#!/usr/bin/env python3
#
# Created on Thu Sep 25 2025 13:00:50
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
from usf.network.layer.spherical.activation import SphericalActivation
from usf.network.layer.spherical.circle_pool import CirclePool
from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.spherical_image import BatchSphericalImage


class Bottleneck(nn.Module):
    """Spherical ResNet V1.5 bottleneck: 1x1 -> spherical conv -> 1x1 with residual connection.

    Optionally down-samples spatial resolution via ``resolution_factor`` and
    projects the skip connection when ``in_channels != out_channels``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        resolution_factor: float | None = None,
        resnet_expansion: int = 4,  # from ResNet Bottleneck V1.5 implementation in torchvision
        *args,
        **kwargs,
    ) -> None:
        """Initialize the ResNet bottleneck block.

        Args:
            in_channels (int): Input channel dimension.
            out_channels (int): Output channel dimension.
            activation (str): Activation function name (must exist in ``torch.nn``).
            resolution_factor (float | None, optional): Spatial resolution change factor.
                None keeps resolution unchanged. Defaults to None.
            resnet_expansion (int, optional): Inverse expansion ratio for the hidden
                dimension (``hidden = out_channels // resnet_expansion``). Defaults to 4.
        """
        super().__init__()

        hidden_channels = int(out_channels // resnet_expansion)

        self.feedforward = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna1",
                        SphericalCBNA(
                            in_channels=in_channels,
                            out_channels=hidden_channels,
                            conv1x1=True,
                            activation=activation,
                        ),
                    ),
                    (
                        "cbna2",
                        SphericalCBNA(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            activation=activation,
                            identical_output_vector=(
                                True if resolution_factor is None else False
                            ),
                            resolution_factor=(
                                1.0 if resolution_factor is None else resolution_factor
                            ),
                            *args,
                            **kwargs,
                        ),
                    ),
                    (
                        "cbna3",
                        SphericalCBNA(
                            in_channels=hidden_channels,
                            out_channels=out_channels,
                            conv1x1=True,
                            activation=False,
                        ),
                    ),
                ]
            )
        )

        if resolution_factor is not None:
            self.value_sampler = ValueSampler(
                {
                    "value_sampler": "radial_basis_function",
                    "value_sampler_config": {
                        "num_in_circle_points": 4,
                        "kernel": "gaussian",
                        "sigma": 0.2,
                        "block_size": kwargs.get("block_size", 8192),
                    },
                    "reject_oo_fov_value": True,
                }
            )

        if in_channels != out_channels:
            self.skip_connection_projection = SphericalCBNA(
                in_channels=in_channels,
                out_channels=out_channels,
                conv1x1=True,
                activation=False,
                *args,
                **kwargs,
            )

        self.activation = SphericalActivation(activation=activation, inplace=True)

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass with residual connection and optional resolution change.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Activated output = feedforward(x) + skip(x).
        """
        skip_connection_batch_spherical_image = batch_spherical_image
        residual_batch_spherical_image = self.feedforward(batch_spherical_image)

        if hasattr(self, "value_sampler"):
            skip_connection_batch_spherical_image = self.value_sampler(
                skip_connection_batch_spherical_image,
                residual_batch_spherical_image.vector,
            )

        if hasattr(self, "skip_connection_projection"):
            skip_connection_batch_spherical_image = self.skip_connection_projection(
                skip_connection_batch_spherical_image
            )

        combined_batch_spherical_image = BatchSphericalImage(
            batch_value=skip_connection_batch_spherical_image.batch_value
            + residual_batch_spherical_image.batch_value,
            vector=skip_connection_batch_spherical_image.vector,
            mask=skip_connection_batch_spherical_image.mask,
        )

        out_batch_spherical_image = self.activation(combined_batch_spherical_image)

        return out_batch_spherical_image


class Stage(nn.Sequential):
    """A ResNet stage: sequence of ``Bottleneck`` blocks where only the first may change resolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        radius_list: list[float],
        resolution_factor: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a ResNet stage.

        Args:
            in_channels (int): Input channels for the first bottleneck.
            out_channels (int): Output channels for all bottlenecks.
            num_bottlenecks (int): Number of bottleneck blocks (must be > 1).
            radius_list (list[float]): Geodesic radii for each bottleneck's spherical conv.
            resolution_factor (float | None, optional): Spatial down-sampling in the first block. Defaults to None.
        """
        assert (
            num_bottlenecks > 1
        ), f"Number of bottlenecks must be more than 1, got {num_bottlenecks}."

        bottlenecks = [
            (
                "bottleneck1",
                Bottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    resolution_factor=resolution_factor,
                    radius=radius_list[0],
                    *args,
                    **kwargs,
                ),
            )
        ]
        for block_index in range(1, num_bottlenecks):
            bottlenecks.append(
                (
                    f"bottleneck{block_index + 1}",
                    Bottleneck(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        radius=radius_list[-1],
                        *args,
                        **kwargs,
                    ),
                )
            )

        super().__init__(OrderedDict(bottlenecks))


class SphericalResNet(nn.Module):
    """Spherical backbone adopting ResNet architecture."""

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
        """Initialize the Spherical ResNet backbone

        Args:
            in_channels (int): Number of channels of the input batch spherical image.
            base_channel (int): Base number of channels used for scaling up in deeper layers.
            activation (str): Activation function type supported by PyTorch.
            location_sampler (str): Type of location sampler for the output vector.
            backend (str): Spherical conv backend name.
            weighting_function_config (dict): Config for weighting function.
            resolution_factor (float): Ratio of output pixels to input pixels for downsampling.
            reject_oo_fov_vector (bool): Whether to reject out-of-field-of-view vectors.
            block_size (int): Block size for pairwise dot operations.
        """
        super().__init__()

        first_conv_weighting_function_config = weighting_function_config.copy()
        if first_conv_weighting_function_config.get("distance") is not None:
            if (
                first_conv_weighting_function_config["distance"]["function"]
                == "discrete"
            ):
                first_conv_weighting_function_config["distance"][
                    "num_slices"
                ] = 7  # analogous to the 7x7 conv2d

        self.stem = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=in_channels,
                            out_channels=base_channel * 2**0,
                            backend=backend,
                            activation=activation,
                            weighting_function_config=first_conv_weighting_function_config,
                            radius=torch.pi * 6 / 600,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                    (
                        "pool",
                        CirclePool(
                            pool_type="max",
                            radius=torch.pi * 3 / 800,
                            resolution_factor=resolution_factor,
                            location_sampler=location_sampler,
                            reject_oo_fov_vector=reject_oo_fov_vector,
                            block_size=block_size,
                        ),
                    ),
                ]
            )
        )

        self.layer1 = Stage(
            in_channels=base_channel * 2**0,
            out_channels=base_channel * 2**2,
            num_bottlenecks=3,
            radius_list=[torch.pi * 6 * 1.1 / 400] * 2,
            backend=backend,
            activation=activation,
            weighting_function_config=weighting_function_config,
            location_sampler=location_sampler,
            reject_oo_fov_vector=reject_oo_fov_vector,
            block_size=block_size,
        )

        self.layer2 = Stage(
            in_channels=base_channel * 2**2,
            out_channels=base_channel * 2**3,
            num_bottlenecks=4,
            radius_list=[torch.pi * 6 * 1.35 / 800, torch.pi * 6 / 180],
            resolution_factor=resolution_factor,
            backend=backend,
            activation=activation,
            weighting_function_config=weighting_function_config,
            location_sampler=location_sampler,
            reject_oo_fov_vector=reject_oo_fov_vector,
            block_size=block_size,
        )

        self.layer3 = Stage(
            in_channels=base_channel * 2**3,
            out_channels=base_channel * 2**4,
            num_bottlenecks=6,
            radius_list=[torch.pi * 6 / 180] * 2,
            backend=backend,
            activation=activation,
            weighting_function_config=weighting_function_config,
            location_sampler=location_sampler,
            reject_oo_fov_vector=reject_oo_fov_vector,
            block_size=block_size,
        )

        self.layer4 = Stage(
            in_channels=base_channel * 2**4,
            out_channels=base_channel * 2**5,
            num_bottlenecks=3,
            radius_list=[torch.pi * 6 / 180] * 2,
            backend=backend,
            activation=activation,
            weighting_function_config=weighting_function_config,
            location_sampler=location_sampler,
            reject_oo_fov_vector=reject_oo_fov_vector,
            block_size=block_size,
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> dict[str, BatchSphericalImage]:
        """Forward pass through stem + 4 ResNet stages.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            dict[str, BatchSphericalImage]: Feature maps keyed ``"layer1"`` … ``"layer4"``.
        """
        stem_batch_spherical_image = self.stem(batch_spherical_image)
        layer1_batch_spherical_image = self.layer1(stem_batch_spherical_image)
        layer2_batch_spherical_image = self.layer2(layer1_batch_spherical_image)
        layer3_batch_spherical_image = self.layer3(layer2_batch_spherical_image)
        layer4_batch_spherical_image = self.layer4(layer3_batch_spherical_image)

        return {
            "layer1": layer1_batch_spherical_image,
            "layer2": layer2_batch_spherical_image,
            "layer3": layer3_batch_spherical_image,
            "layer4": layer4_batch_spherical_image,
        }
