#!/usr/bin/env python3
#
# Created on Fri Nov 22 2024 16:44:51
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
from typing import Any

import numpy as np
import torch
from cachetools import LFUCache
from multimethod import multimethod

from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.nearest_neighbor import nearest_point
from usf.utils.spherical_image import BatchSphericalImage
from usf.utils.torch_numpy import array_hash, to_numpy, to_torch


class NearestNeighbor(ValueSampler):
    """Value sampler using inverse-distance-weighted k-nearest-neighbor interpolation."""

    # class level cache shared among instances, need to guarantee uniqueness
    index_angle_cache = LFUCache(maxsize=500)

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize nearest-neighbor value sampler.

        Args:
            config (dict): Must contain ``"value_sampler_config"`` with optional
                keys ``num_neighbors`` (int, default 4),
                ``softmax_temperature`` (float, default 0.001),
                ``eps`` (float, default 1e-10).
        """
        super().__init__(config, *args, **kwargs)
        self.value_sampler_config: dict[str, Any] = config.get(
            "value_sampler_config", {}
        )
        self.num_neighbors: int = self.value_sampler_config.get("num_neighbors", 4)
        self.softmax_temperature: float = self.value_sampler_config.get(
            "softmax_temperature", 0.001
        )
        self.eps: float = self.value_sampler_config.get("eps", 1e-10)

    def extra_args(self) -> str:
        """overridden function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        return (
            f"num_neighbors={self.num_neighbors}, "
            f"softmax_temperature={self.softmax_temperature}"
        )

    @multimethod
    @torch.no_grad()
    def sample_value(  # type: ignore
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: np.ndarray,
        *,
        num_neighbors: int | None = None,
        softmax_temperature: float | None = None,
        eps: float | None = None,
    ) -> np.ndarray:
        """
        Sample pixel values from spherical_image using a softmax weighting of the num_neighbors nearest neighbors.

        Args:
            batch_spherical_image (BatchSphericalImage): The existing color data and their 3D or spherical coords.
            sample_vector (np.ndarray): shape (M,3) or (M,2), new sample locations on sphere.
            num_neighbors (int | None, optional): number of neighbors to use in interpolation. Defaults to None.
            softmax_temperature (float | None, optional): softmax temperature "T", smaller T => sharper weighting. Defaults to None.
            eps (float | None, optional): small offset for numeric safety. Defaults to None.

        Returns:
            np.ndarray
        """
        assert isinstance(
            batch_spherical_image.batch_value, np.ndarray
        ), "Expect input BatchSphericalImage to use np.ndarray backend, same to sample_vector"

        B, I, C = batch_spherical_image.batch_value.shape

        # 1) Extract old data (assuming we already stored them in 3D unit vectors)
        image_vector = batch_spherical_image.vector  # shape (N,3)
        sample_vector.shape[0]

        num_neighbors = self.num_neighbors if num_neighbors is None else num_neighbors
        softmax_temperature = (
            self.softmax_temperature
            if softmax_temperature is None
            else softmax_temperature
        )
        eps = self.eps if eps is None else eps

        # cache the nn index for the input image_vector
        input_output_vector_hash = (
            f"{array_hash(image_vector)}_{array_hash(sample_vector)}"
        )
        cache_key = (
            f"nearest_neighbor_{input_output_vector_hash}_{num_neighbors}".replace(
                ".", "dot"
            )
        )
        index_buffer_name = f"nearest_neighbor_index_{input_output_vector_hash}_{num_neighbors}".replace(
            ".", "dot"
        )
        angle_buffer_name = f"nearest_neighbor_angle_{input_output_vector_hash}_{num_neighbors}".replace(
            ".", "dot"
        )

        # Get cache and buffer (if they exist)
        index_buffer_entry = getattr(self, index_buffer_name, None)
        angle_buffer_entry = getattr(self, angle_buffer_name, None)

        # If one is missing, recover it from the other
        if cache_key not in self.index_angle_cache and index_buffer_entry is not None:
            neighbor_indices, angles = index_buffer_entry, angle_buffer_entry

            self.index_angle_cache[cache_key] = (
                neighbor_indices,
                angles,
            )
        elif cache_key in self.index_angle_cache and index_buffer_entry is None:
            neighbor_indices, angles = self.index_angle_cache[cache_key]

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, neighbor_indices)
                self.register_buffer(angle_buffer_name, angles)
        elif cache_key not in self.index_angle_cache and index_buffer_entry is None:
            # Both are missing, so compute and cache the result
            neighbor_indices, angles = nearest_point(
                sample_vector,
                image_vector,
                num_neighbors,
            )

            neighbor_indices, angles = to_torch(neighbor_indices), to_torch(angles)

            self.index_angle_cache[cache_key] = neighbor_indices, angles

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, neighbor_indices)
                self.register_buffer(angle_buffer_name, angles)
        else:
            # no cache missing
            neighbor_indices, angles = index_buffer_entry, angle_buffer_entry

        (
            neighbor_indices,
            angles,
        ) = (
            to_numpy(neighbor_indices),
            to_numpy(angles),
        )

        # 4) Softmax weighting: w_j = exp( - angles_j / T ) / Z
        #    If T is small => "sharper" weighting on the nearest neighbor.
        #    angles in [0, π]
        logits = -angles / (softmax_temperature + eps)  # shape (M,num_neighbors)
        # exponentiation
        raw_weights = np.exp(logits)  # shape (M,num_neighbors)
        weight_sum = np.sum(raw_weights, axis=1, keepdims=True) + eps
        weights = raw_weights / weight_sum  # shape (M,num_neighbors)
        # Broadcast weights (shape (M, num_neighbors)) to the batch: (B, M, num_neighbors, 1)
        weights_expand = weights[None, :, :, None]

        # Gather neighbor colors from each image in the batch.
        # batch_value: (B, N, C); neighbor_indices: (M, num_neighbors)
        # Expand neighbor_indices to (B, M, num_neighbors) by repeating along the batch dimension.
        neighbor_indices_batched = np.tile(neighbor_indices[None, :, :], (B, 1, 1))

        # Use advanced indexing to gather colors.
        gathered_colors = batch_spherical_image.batch_value[
            np.arange(B)[:, None, None], neighbor_indices_batched
        ]  # shape (B, M, num_neighbors, C)

        color_contrib = (
            weights_expand * gathered_colors
        )  # shape (B, M, num_neighbors, C)

        # Sum over the num_neighbors neighbors to get the sampled color for each sample point.
        sampled_colors = np.sum(color_contrib, axis=2)  # shape (B, M, C)

        return sampled_colors

    @multimethod
    # @torch.compiler.disable(recursive=False)
    def sample_value(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: torch.Tensor,
        *,
        num_neighbors: int | None = None,
        softmax_temperature: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """
        Sample pixel values from spherical_image using a softmax weighting of the num_neighbors nearest neighbors.

        Args:
            batch_spherical_image (BatchSphericalImage): The existing color data and their 3D or spherical coords.
            sample_vector (torch.Tensor): shape (M,3) or (M,2), new sample locations on sphere.
            num_neighbors (int | None, optional): number of neighbors to use in interpolation. Defaults to None.
            softmax_temperature (float | None, optional): softmax temperature "T", smaller T => sharper weighting. Defaults to None.
            eps (float | None, optional): small offset for numeric safety. Defaults to None.

        Returns:
            torch.Tensor
        """
        assert isinstance(
            batch_spherical_image.batch_value, torch.Tensor
        ), "Expect input BatchSphericalImage to use torch.Tensor backend, same to sample_vector"

        # Get device and shapes from the input batch.
        _device = batch_spherical_image.batch_value.device
        B, I, C = batch_spherical_image.batch_value.shape

        # 1) Extract old data (assuming we already stored them in 3D unit vectors)
        image_vector = batch_spherical_image.vector  # shape (N,3)
        sample_vector.shape[0]

        num_neighbors = self.num_neighbors if num_neighbors is None else num_neighbors
        softmax_temperature = (
            self.softmax_temperature
            if softmax_temperature is None
            else softmax_temperature
        )
        eps = self.eps if eps is None else eps

        # cache the neighbor_indices and angle given input and output vector
        input_output_vector_hash = (
            f"{array_hash(image_vector)}_{array_hash(sample_vector)}"
        )
        cache_key = (
            f"nearest_neighbor_{input_output_vector_hash}_{num_neighbors}".replace(
                ".", "dot"
            )
        )
        index_buffer_name = f"nearest_neighbor_index_{input_output_vector_hash}_{num_neighbors}".replace(
            ".", "dot"
        )
        angle_buffer_name = f"nearest_neighbor_angle_{input_output_vector_hash}_{num_neighbors}".replace(
            ".", "dot"
        )

        # Get cache and buffer (if they exist)
        index_buffer_entry = getattr(self, index_buffer_name, None)
        angle_buffer_entry = getattr(self, angle_buffer_name, None)

        # If one is missing, recover it from the other
        if cache_key not in self.index_angle_cache and index_buffer_entry is not None:
            neighbor_indices, angles = index_buffer_entry, angle_buffer_entry

            self.index_angle_cache[cache_key] = (
                neighbor_indices,
                angles,
            )
        elif cache_key in self.index_angle_cache and index_buffer_entry is None:
            neighbor_indices, angles = self.index_angle_cache[cache_key]

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, neighbor_indices)
                self.register_buffer(angle_buffer_name, angles)
        elif cache_key not in self.index_angle_cache and index_buffer_entry is None:
            # Both are missing, so compute and cache the result
            neighbor_indices, angles = nearest_point(
                sample_vector,
                image_vector,
                num_neighbors,
            )

            self.index_angle_cache[cache_key] = neighbor_indices, angles

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, neighbor_indices)
                self.register_buffer(angle_buffer_name, angles)
        else:
            # no cache missing
            neighbor_indices, angles = index_buffer_entry, angle_buffer_entry

        (
            neighbor_indices,
            angles,
        ) = (
            to_torch(neighbor_indices, device=_device),
            to_torch(angles, device=_device),
        )

        with torch.no_grad():
            # 4) Softmax weighting: w_j = exp( - angles_j / T ) / Z
            #    If T is small => "sharper" weighting on the nearest neighbor.
            #    angles in [0, π]
            logits = -angles / (softmax_temperature + eps)  # shape (M, num_neighbors)
            # exponentiation
            raw_weights = torch.exp(logits)  # shape (M, num_neighbors)
            weight_sum = torch.sum(raw_weights, dim=1, keepdims=True) + eps
            weights = raw_weights / weight_sum  # shape (M, num_neighbors)
            # Expand weights for broadcasting: (M, num_neighbors) -> (B, M, num_neighbors, 1)
            weights_expand = weights.unsqueeze(0).unsqueeze(-1)

            # batch_value: (B, N, C), neighbor_indices: (M, num_neighbors) -> need shape (B, sample_vector.shape[0], num_neighbors)
            neighbor_indices_batched = neighbor_indices.unsqueeze(0).expand(
                B, sample_vector.shape[0], num_neighbors
            )
            batch_idx = (
                torch.arange(B, device=sample_vector.device).unsqueeze(1).unsqueeze(2)
            )  # shape (B,1,1)

        # Gather: result will be (B, M, num_neighbors, C)
        gathered_colors = batch_spherical_image.batch_value[
            batch_idx, neighbor_indices_batched
        ]
        color_contrib = weights_expand * gathered_colors  # (B, M, num_neighbors, C)
        sampled_colors = torch.sum(color_contrib, dim=2)  # (B, M, C)

        return sampled_colors


register_cache(
    "ValueSampler.NearestNeighbor.index_angle_cache", NearestNeighbor.index_angle_cache
)
