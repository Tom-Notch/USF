#!/usr/bin/env python3
#
# Created on Wed Sep 10 2025 19:39:34
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class PlanarDeepLabV3(nn.Module):
    """Planar backbone adopting Deep Lab V3 architecture."""

    def __init__(
        self,
        load_pretrain: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the DeepLabV3 backbone.

        Args:
            load_pretrain (bool, optional): Load ImageNet-pretrained weights. Defaults to True.
        """
        super().__init__()

        self.model = deeplabv3_resnet50(
            weights=DeepLabV3_ResNet50_Weights.DEFAULT if load_pretrain else None
        )
        del self.model.classifier[-1]  # remove last linear layer
        self.model.aux_classifier = None  # remove auxiliary loss branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deep features via DeepLabV3-ResNet50.

        Args:
            x (torch.Tensor): Input of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Feature map from the classifier head (B, C, H', W').
        """
        return self.model(x)["out"]
