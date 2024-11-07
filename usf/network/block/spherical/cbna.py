#!/usr/bin/env python3
#
# Created on Mon Feb 17 2025 22:14:14
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.layer.spherical.activation import SphericalActivation
from usf.network.layer.spherical.batchnorm import SphericalBatchNorm
from usf.network.layer.spherical.conv1x1 import SphericalConv1x1
from usf.network.layer.spherical.generic_spherical_cnn import GenericSphericalConv
from usf.network.layer.spherical.interpolation import SphericalInterpolation


class SphericalCBNA(nn.Sequential):
    """Spherical Convolution + Batch Normalization + Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str | bool,
        radius: float | None = None,
        backend: str = "circle",
        weighting_function_config: dict = {
            "distance": {
                "function": "continuous",
                "activation": "ReLU",
                "hidden_dims": [8, 8],
            }
        },
        activation_config: dict = {"inplace": True},
        conv1x1: bool = False,
        reject_oo_fov_vector: bool = False,
        interpolation: dict | None = None,
        *args,
        **kwargs,
    ):
        """
        Initializes a SphericalCBNA block which performs a convolution (either a 1x1 convolution
        or a circle convolution), followed by batch normalization and an activation function. The block is
        implemented as an nn.Sequential container with three layers: a convolution layer, a batch normalization
        layer, and an activation layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str | bool): Activation function name, or False to disable.
            radius (float | None, optional): Geodesic radius (in radians) of the kernel. Defaults to None.
            backend (str, optional): Spherical conv backend name. Defaults to ``"circle"``.
            weighting_function_config (dict, optional): Config for weighting function.
            activation_config (dict, optional): Kwargs passed to the activation constructor. Defaults to ``{"inplace": True}``.
            conv1x1 (bool, optional): If True, uses SphericalConv1x1 instead of full conv. Defaults to False.
            reject_oo_fov_vector (bool, optional): Whether to reject out-of-FoV vectors. Defaults to False.
            interpolation (dict | None, optional): Config for optional interpolation layer appended after activation. Defaults to None.
        """
        layer_list: list[tuple[str, nn.Module]] = []

        if interpolation:
            layer_list.append(
                (
                    "interpolation",
                    SphericalInterpolation(
                        config=(interpolation if interpolation else None)
                    ),
                )
            )
            layer_list.append(
                (
                    "conv1x1",
                    SphericalConv1x1(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=False,
                    ),
                )
            )
        elif conv1x1 is True:
            layer_list.append(
                (
                    "conv1x1",
                    SphericalConv1x1(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=False,
                    ),
                )
            )
        else:
            assert radius is not None, "radius is required for spherical convolution"
            layer_list.append(
                (
                    "conv",
                    GenericSphericalConv(
                        backend=backend,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        weighting_function_config=weighting_function_config,
                        radius=radius,
                        reject_oo_fov_vector=reject_oo_fov_vector,
                        bias=False,
                        *args,
                        **kwargs,
                    ),
                )
            )

        layer_list.append(("batchnorm", SphericalBatchNorm(num_features=out_channels)))
        layer_list.append(
            (
                "activation",
                SphericalActivation(activation=activation, **activation_config),
            )
        )

        super().__init__(OrderedDict(layer_list))
