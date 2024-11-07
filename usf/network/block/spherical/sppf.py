#!/usr/bin/env python3
#
# Created on Sun Feb 23 2025 13:53:42
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn

from usf.network.block.spherical.cbna import SphericalCBNA
from usf.network.layer.spherical.circle_pool import CirclePool
from usf.network.layer.spherical.global_pool import SphericalGlobalPool
from usf.utils.spherical_image import BatchSphericalImage, concatenate


class SphericalSPPF(nn.Module):
    """
    Spherical Spatial Pyramid Pooling - Fast (SPPF) layer.

    This layer processes a BatchSphericalImage as follows:
      1. Applies a 1x1 convolution (cbna1) to reduce the input channels to hidden_channels,
         where hidden_channels = int(in_channels * expansion). (Default expansion=0.5 mimics c1//2.)
      2. Applies the CirclePool operator repeatedly (num_pooling_layers times).
      3. Concatenates the original output from cbna1 with each pooled output using the provided
         concatenate function.
      4. Fuses the concatenated features with a final 1x1 convolution (cbna2) to produce the output.

    Additional parameters for CirclePool (beyond pool_type) are forwarded via *args/**kwargs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        num_pooling_layers: int,
        pool_type: str = "max",
        radius: float = torch.pi / 80,
        global_pool: bool = False,
        expansion: float = 0.5,
        *args,
        **kwargs,
    ):
        """Initialize the SphericalSPPF block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (str): Activation function for SphericalCBNA.
            num_pooling_layers (int): Number of CirclePool operations to perform.
            pool_type (str, optional): Type of pooling ("max", "min", or "mean"). Defaults to "max".
            radius (float, optional): Geodesic radius controlling size of the pool circle. Defaults to torch.pi / 80.
            global_pool (bool, optional): Whether to include global pooling as an additional branch. Defaults to False.
            expansion (float, optional): Expansion factor to compute hidden_channels.
                                         hidden_channels = int(in_channels * expansion). Defaults to 0.5.
        """
        super().__init__()
        hidden_channels = int(in_channels * expansion)
        # First 1x1 conv: reduces channels to hidden_channels.
        self.cbna1 = SphericalCBNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            conv1x1=True,
            *args,
            **kwargs,
        )
        # CirclePool operator with pool_type; additional parameters are forwarded.
        self.circle_pool = CirclePool(
            pool_type=pool_type,
            radius=radius,
            identical_output_vector=True,
            *args,
            **kwargs,
        )
        if global_pool:
            self.global_pool = SphericalGlobalPool(pool_type=pool_type)
        self.num_pooling_layers = num_pooling_layers
        # Final 1x1 conv: fuses concatenated outputs.
        # There will be (1 + num_pooling_layers) outputs concatenated, each with hidden_channels.
        self.cbna2 = SphericalCBNA(
            in_channels=hidden_channels
            * (num_pooling_layers + (2 if global_pool else 1)),
            out_channels=out_channels,
            activation=activation,
            conv1x1=True,
            *args,
            **kwargs,
        )

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: Number of pooling layers.
        """
        return f"num_pooling_layers={self.num_pooling_layers}"

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Forward pass: cascaded circle pooling at the same radius, then concatenate.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Fused multi-scale pooled features.
        """
        branches = [self.cbna1(batch_spherical_image)]
        for _ in range(self.num_pooling_layers):
            branches.append(self.circle_pool(branches[-1]))
        if hasattr(self, "global_pool"):
            branches.append(self.global_pool(branches[0]))
        return self.cbna2(concatenate(branches))
