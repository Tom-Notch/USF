#!/usr/bin/env python3
#
# Created on Tue Oct 21 2025 12:57:13
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from collections import OrderedDict

import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.layer.spherical.dropout import SphericalDropout
from usf.network.layer.spherical.global_pool import SphericalGlobalPool
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) for spherical DeepLabV3."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        radius_list: list[float],
        global_pool: bool = False,
        *args,
        **kwargs,
    ):
        """Initialize the Spherical ASPP module

        Args:
            in_channels (int): input channel
            out_channels (int): output channel
            activation (str): activation function type supported by PyTorch
            radius_list (list[float]): list of geodesic radii for each ASPP branch
            global_pool (bool): whether to include global pooling as an additional branch
        """
        super().__init__()

        cbnas: list[tuple[str, nn.Module]] = [
            (
                "conv1x1",
                SphericalCBNA(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv1x1=True,
                    activation=activation,
                ),
            )
        ]

        for i, radius in enumerate(radius_list):
            cbnas.append(
                (
                    f"ASPPConv{i + 1}",
                    SphericalCBNA(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        activation=activation,
                        radius=radius,
                        identical_output_vector=True,
                        *args,
                        **kwargs,
                    ),
                )
            )

        if global_pool:
            cbnas.append(
                (
                    "ASPPPooling",
                    nn.Sequential(
                        OrderedDict(
                            (
                                (
                                    "global_pool",
                                    SphericalGlobalPool(pool_type="mean"),
                                ),
                                (
                                    "cbna",
                                    SphericalCBNA(
                                        in_channels=in_channels,
                                        out_channels=out_channels,
                                        conv1x1=True,
                                        activation=activation,
                                    ),
                                ),
                            )
                        )
                    ),
                )
            )

        self.multi_scale_branch = nn.Sequential(OrderedDict(cbnas))

        self.project = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cbna",
                        SphericalCBNA(
                            in_channels=out_channels * len(cbnas),
                            out_channels=out_channels,
                            activation=activation,
                            conv1x1=True,
                        ),
                    ),
                    ("dropout", SphericalDropout(p=0.5)),
                ]
            )
        )

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: apply multi-scale convolutions in parallel, concatenate, project.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Projected multi-scale features.
        """
        branches = [conv(batch_spherical_image) for conv in self.multi_scale_branch]
        output_batch_spherical_image = self.project(concatenate(branches))

        return output_batch_spherical_image
