#!/usr/bin/env python3
#
# Created on Sun Feb 16 2025 16:20:11
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch.nn as nn

from usf.utils.spherical_image import BatchSphericalImage


class SphericalActivation(nn.Module):
    """Apply activation only on the batch_value"""

    def __init__(self, activation: str | bool, *args, **kwargs):
        """
        Spherical Activation Module

        Args:
            activation (str | bool): Name of the activation function (must exist in `torch.nn`).
                                    Or False to disable.
        """
        super().__init__()

        self.activation = activation

        if self.activation:
            if getattr(nn, self.activation, None) is not None:
                self.act = getattr(nn, self.activation)(*args, **kwargs)
            elif isinstance(self.activation, nn.Module):
                self.act = self.activation
            else:
                raise ValueError(f"Unsupported activation type: {self.activation}")

    def extra_repr(self) -> str:
        """Return compact string summary.

        Returns:
            str: ``"activation=False"`` when disabled, empty otherwise.
        """
        return f"activation=False" if not self.activation else ""

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Apply activation to ``batch_value``, passing geometry through.

        Args:
            batch_spherical_image (BatchSphericalImage): Input.

        Returns:
            BatchSphericalImage: Activated output (or passthrough if disabled).
        """
        return (
            BatchSphericalImage(
                batch_value=self.act(batch_spherical_image.batch_value),
                vector=batch_spherical_image.vector,
            )
            if self.activation
            else batch_spherical_image
        )
