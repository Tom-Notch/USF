#!/usr/bin/env python3
#
# Created on Fri Feb 21 2025 18:52:18
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cachetools import LFUCache
from matplotlib import cm
from opt_einsum import contract
from torch.utils.checkpoint import checkpoint

from usf.network.layer.spherical.backend.weighting_function.weighting_function import (
    MultiBranchWeightingFunction,
)
from usf.network.layer.spherical.generic_spherical_cnn import GenericSphericalConv
from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.spherical import colorize_locations, relative_direction, ripple_sort
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import (
    array_hash,
    array_size_MiB,
    copy_or_clone,
    module_weights_hash,
    to_numpy,
    to_torch,
)
from usf.visualization.spherical_layer import sample_circles


class SphericalConv(GenericSphericalConv):
    """Spherical Convolution with generic separate weighting function for geodesic distance and 2D direction"""

    # class level cache shared among instances, need to guarantee uniqueness
    output_vector_cache = LFUCache(maxsize=500)
    collection_matrix_cache = LFUCache(maxsize=500)

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        radius: float,
        weighting_function_config: dict[str, dict[str, Any]],
        identical_output_vector: bool = False,
        resolution_factor: float = 1.0,
        location_sampler: str = "icosahedron",
        reject_oo_fov_vector: bool = True,
        bias: bool = True,
        block_size: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        backend: str = "spherical",
    ):
        """continuous ripple convolution on spherical signals (S2)

        Args:
            in_channels (int): number of input channels of SphericalImage.value
            out_channels (int): number of output channels of SphericalImage.value
            radius (float): in radians, geodesic radius of the kernel at an output location, analogous to kernel size
            weighting_function_config (dict[str, dict[str, Any]]): Weighting function config.
            identical_output_vector (bool, optional): whether output_vector is forcefully set to just input_vector
            resolution_factor (float, optional): factor to in/decrease the resolution of the input SphericalImage
                                       # output_pixels \approx resolution_factor * # input_pixels
            location_sampler (str, optional): which LocationSample to use for novel pixel location.
                                     Defaults to icosahedron.
            reject_oo_fov_vector (bool, optional): whether to use panorama mode for location sampler, turning on will disable out-of-view vector rejection. Defaults to False.
            bias (bool, optional): bias applied onto each output channels. Defaults to True.
            block_size (int, optional): the size of a block for large pairwise dot, each block is chunked into block_size x block_size. Can enlarge depending on how big your (V)RAM is. Defaults to 2048.
            device (torch.device | None, optional): torch device. Defaults to None.
            dtype (torch.dtype | None, optional): torch data type. Defaults to None.
            backend (str, optional): to swallow backend kwarg from parent class initialization. Defaults to "spherical".
        """
        assert (
            backend == "spherical"
        ), f"wrong backend instantiated, got backend = {backend}"
        super().__init__()

        assert in_channels > 0, "in_channels must be > 0"
        assert out_channels > 0, "out_channels must be > 0"
        assert 0.0 < radius <= torch.pi, f"radius {radius} must be in (0, π]"
        assert resolution_factor > 0.0, "resolution_factor must be > 0.0"
        assert block_size > 0, "block_size must be positive"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radius = radius

        # fill in missing fields for weighting function configs
        for subconfig in weighting_function_config.values():
            subconfig["in_channels"] = 1
            subconfig["out_channels"] = 1
            subconfig["num_functions"] = self.in_channels * self.out_channels

        distance_config = weighting_function_config.get("distance", None)
        if distance_config is not None:
            function_type = distance_config.get("function", None)
            if function_type == "discrete":
                # remember distance is normalized from [0, radius] to [0.0, 1.0]
                distance_config["value_min"] = 0.0
                distance_config["value_max"] = 1.0

        direction_config = weighting_function_config.get("direction", None)
        if direction_config is not None:
            function_type = direction_config.get("function", None)
            if function_type == "discrete":
                # remember distance is normalized from [-pi, pi] to [-1.0, 1.0]
                direction_config["value_min"] = -1.0
                direction_config["value_max"] = 1.0
            elif function_type == "continuous":
                direction_config["fourier_include_x"] = False

        self.weighting_function = MultiBranchWeightingFunction(
            weighting_function_config
        )

        self.identical_output_vector = identical_output_vector

        self.resolution_factor = resolution_factor
        self.location_sampler_type = location_sampler
        self.reject_oo_fov_vector = reject_oo_fov_vector
        if not self.identical_output_vector:
            self.location_sampler = LocationSampler(
                config={
                    "location_sampler": self.location_sampler_type,
                    "resolution_factor": self.resolution_factor,
                    "reject_oo_fov_vector": self.reject_oo_fov_vector,
                }
            )

        self.block_size = block_size

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self.to(device=device, dtype=dtype)

    def reset_parameters(self) -> SphericalConv:
        """Reinitialize bias (uniform, scaled by kernel radius).

        Returns:
            SphericalConv: ``self`` for chaining.
        """
        if isinstance(getattr(self, "bias", None), nn.Parameter):
            # ! magic numbers, reduce variance when aggregating many inputs
            bound = 1 / math.sqrt(self.in_channels * (self.radius / math.pi))
            nn.init.uniform_(self.bias, -bound, bound)

        return self

    def extra_repr(self) -> str:
        """For printing Module information

        Returns:
            str: args and kwargs of the Module
        """
        extra_repr = (
            f"{self.in_channels}" f", {self.out_channels}" f", radius={self.radius:.5f}"
        )

        if isinstance(getattr(self, "override_output_vector", None), torch.Tensor):
            extra_repr += ", override_output_vector=True"
        elif self.identical_output_vector:
            extra_repr += ", identical_output_vector=True"

        extra_repr += (
            f", bias={getattr(self, 'bias', None) is not None}"
            f", block_size={self.block_size}"
        )

        return extra_repr

    def _get_output_vector(self, input_vector: torch.Tensor) -> torch.Tensor:
        _device, _dtype = self.device, self.dtype

        if isinstance(getattr(self, "override_output_vector", None), torch.Tensor):
            output_vector = self.override_output_vector
        elif self.identical_output_vector:
            output_vector = input_vector
        else:
            # class level uniqueness guaranteed
            # not using ripple_sort since sorting the value may break differentiability
            input_vector_hash = (
                array_hash(input_vector)
                + "_"
                + str(self.location_sampler_type)
                + "_"
                + str(self.reject_oo_fov_vector)
                + "_"
                + str(self.resolution_factor).replace(
                    ".", "dot"
                )  # buffer name doesn't allow '.'
            )
            buffer_name = f"output_vector_{input_vector_hash}"

            # Get buffer (if it exists)
            buffer_entry = getattr(self, buffer_name, None)

            # If one is missing, recover it from the other
            if (
                input_vector_hash not in self.output_vector_cache
                and buffer_entry is not None
            ):
                output_vector = buffer_entry
                self.output_vector_cache[input_vector_hash] = output_vector
            elif input_vector_hash in self.output_vector_cache and buffer_entry is None:
                output_vector = self.output_vector_cache[input_vector_hash]
                if is_cache_enabled():
                    self.register_buffer(buffer_name, output_vector)
            elif (
                input_vector_hash not in self.output_vector_cache
                and buffer_entry is None
            ):
                # Both are missing, so compute and cache the result
                with torch.no_grad():
                    output_vector = self.location_sampler(
                        input_vector, resolution_factor=self.resolution_factor
                    ).to(device=_device, dtype=_dtype)
                self.output_vector_cache[input_vector_hash] = output_vector
                if is_cache_enabled():
                    self.register_buffer(buffer_name, output_vector)
            else:
                output_vector = buffer_entry

        return output_vector.to(device=_device, dtype=_dtype)

    @torch.no_grad()
    def _block_collection_matrix(
        self,
        output_vectors: torch.Tensor,
        input_vectors: torch.Tensor,
        block_size: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the padded collection matrix in a ring-based style.

        For each output vector (of shape (num_output_pixels, 3)) and each input vector
        (of shape (num_input_pixels, 3)), if the dot product is above the cosine_threshold
        (i.e. within the influence range defined by radius), record the input index and
        compute the geodesic distance (via arccos).

        Returns four tensors:
        - padded_indices: candidate input indices, shape (num_output_pixels, num_max_candidates)
        - padded_normalized_distances: candidate geodesic distances, shape (num_output_pixels, num_max_candidates, 1), Value range in [0.0, 1.0]
        - padded_normalized_directions: candidate directions, shape (num_output_pixels, num_max_candidates, 1) Value range in [-1.0, 1.0]
        - mask: boolean mask indicating valid candidate positions, shape (num_output_pixels, num_max_candidates)
        """
        assert (
            output_vectors.shape[-1] == input_vectors.shape[-1]
        ), "Last dimension must match"
        _device, _dtype = self.device, self.dtype

        # Unify device and dtype
        output_vectors = output_vectors.to(device=_device, dtype=_dtype)
        input_vectors = input_vectors.to(device=_device, dtype=_dtype)

        num_output_pixels = output_vectors.shape[0]
        input_vectors.shape[0]

        cosine_threshold = torch.cos(
            torch.tensor(self.radius, device=_device, dtype=_dtype)
        )

        membership_indices_list: list[torch.Tensor] = []
        distance_values_list: list[torch.Tensor] = []
        direction_values_list: list[torch.Tensor] = []

        # Chunk
        output_chunks = torch.split(output_vectors, block_size, dim=0)
        input_chunks = torch.split(input_vectors, block_size, dim=0)

        output_offset = 0
        for output_chunk in output_chunks:
            input_offset = 0
            for input_chunk in input_chunks:
                # shape: (chunk_output_pixels, chunk_input_pixels)
                dot_block = torch.clamp(output_chunk @ input_chunk.T, -1.0, 1.0)
                valid_mask_block = dot_block >= cosine_threshold
                if valid_mask_block.sum() > 0:
                    # shape: (chunk_output_pixels, chunk_input_pixels)
                    distance_block = torch.acos(dot_block)
                    # shape: (num_valid, 2)
                    valid_indices = torch.nonzero(valid_mask_block)

                    valid_output_vector = output_chunk[valid_indices[:, 0]]
                    valid_input_vector = input_chunk[valid_indices[:, 1]]
                    direction_values_list.append(
                        relative_direction(
                            valid_output_vector, valid_input_vector
                        ).unsqueeze(1)
                    )

                    valid_indices[:, 0] += output_offset
                    valid_indices[:, 1] += input_offset
                    membership_indices_list.append(valid_indices)

                    distance_values_list.append(
                        distance_block[valid_mask_block].unsqueeze(1)
                    )
                input_offset += input_chunk.shape[0]

            output_offset += output_chunk.shape[0]

        if len(membership_indices_list) == 0:
            padded_indices = torch.zeros(
                (num_output_pixels, 1),
                dtype=torch.int64,
                device=_device,
            )
            padded_normalized_distances = torch.zeros(
                (num_output_pixels, 1),
                dtype=_dtype,
                device=_device,
            )
            padded_normalized_directions = torch.zeros(
                (num_output_pixels, 1),
                dtype=_dtype,
                device=_device,
            )
            mask = torch.zeros(
                (num_output_pixels, 1),
                dtype=torch.bool,
                device=_device,
            )
            return (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            )

        # shape: (num_valid_total, 2)
        membership_indices = torch.cat(membership_indices_list, dim=0)
        # shape: (num_valid_total, 1)
        distance_values = torch.cat(distance_values_list, dim=0)
        # shape: (num_valid_total, 1)
        direction_values = torch.cat(direction_values_list, dim=0)

        # Sort by output index
        _, sort_order = torch.sort(membership_indices[:, 0])
        membership_indices = membership_indices[sort_order]
        distance_values = distance_values[sort_order]
        direction_values = direction_values[sort_order]

        # Compute per-output candidate counts
        unique_output_indices, counts = torch.unique_consecutive(
            membership_indices[:, 0], return_counts=True
        )
        num_max_candidates = int(counts.max().item())

        # Initialize padded indices, geodesic distances, and mask
        padded_indices = torch.zeros(
            (num_output_pixels, num_max_candidates),
            dtype=torch.int64,
            device=_device,
        )
        padded_geodesic_distances = torch.zeros(
            (num_output_pixels, num_max_candidates),
            dtype=_dtype,
            device=_device,
        )
        padded_directions = torch.zeros(
            (num_output_pixels, num_max_candidates),
            dtype=_dtype,
            device=_device,
        )
        mask = torch.zeros(
            (num_output_pixels, num_max_candidates),
            dtype=torch.bool,
            device=_device,
        )

        # Compute cumulative counts to segment indices
        cumulative_counts = torch.cat(
            (torch.tensor([0], device=_device), counts.cumsum(dim=0))
        )

        for i in range(unique_output_indices.shape[0]):
            out_idx = unique_output_indices[i].item()  # Output index
            start_idx = cumulative_counts[i].item()
            end_idx = cumulative_counts[i + 1].item()
            # Length of candidates for this output index
            num_candidates = end_idx - start_idx

            padded_indices[out_idx, :num_candidates] = membership_indices[
                start_idx:end_idx, 1
            ]
            padded_geodesic_distances[out_idx, :num_candidates] = distance_values[
                start_idx:end_idx, 0
            ]
            padded_directions[out_idx, :num_candidates] = direction_values[
                start_idx:end_idx, 0
            ]
            mask[out_idx, :num_candidates] = True

        # Normalize geodesic distances to [0, 1] based on radius
        padded_normalized_distances = padded_geodesic_distances / self.radius

        # Normalize direction to [-1, 1]
        padded_normalized_directions = padded_directions / torch.pi

        return (
            padded_indices,
            padded_normalized_distances.unsqueeze(-1),
            padded_normalized_directions.unsqueeze(-1),
            mask,
        )

    def _get_collection_matrix(
        self, input_vectors: torch.Tensor, output_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the cached collection matrix if available; otherwise computes it.

        The collection matrix consists of tensors:
        - padded_indices, shape (num_output_pixels, num_max_candidates)
        - padded_normalized_distances, shape (num_output_pixels, num_max_candidates, ), value range in [0.0, 1.0]
        - padded_normalized_directions, shape (num_output_pixels, num_max_candidates, ), value range in [-1.0, 1.0]
        - mask, shape (num_output_pixels, num_max_candidates)

        These are computed based on input_vectors and output_vectors.
        """
        _device = self.device

        input_output_hash = (
            array_hash(torch.vstack((input_vectors, output_vectors)))
            + "_"
            + str(self.radius).replace(".", "dot")
        )
        padded_indices_buffer_name = f"padded_indices_{input_output_hash}"
        padded_normalized_distances_buffer_name = (
            f"padded_normalized_distances_{input_output_hash}"
        )
        padded_normalized_directions_buffer_name = (
            f"padded_normalized_directions_{input_output_hash}"
        )
        mask_buffer_name = f"mask_{input_output_hash}"

        # Get buffers (if they exist)
        padded_indices_buffer_entry = getattr(self, padded_indices_buffer_name, None)
        padded_normalized_distances_buffer_entry = getattr(
            self, padded_normalized_distances_buffer_name, None
        )
        padded_normalized_directions_buffer_entry = getattr(
            self, padded_normalized_directions_buffer_name, None
        )
        mask_buffer_entry = getattr(self, mask_buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_output_hash not in self.collection_matrix_cache
            and padded_indices_buffer_entry is not None
        ):
            (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            ) = (
                padded_indices_buffer_entry,
                padded_normalized_distances_buffer_entry,
                padded_normalized_directions_buffer_entry,
                mask_buffer_entry,
            )
            self.collection_matrix_cache[input_output_hash] = (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            )
        elif (
            input_output_hash in self.collection_matrix_cache
            and padded_indices_buffer_entry is None
        ):
            (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            ) = self.collection_matrix_cache[input_output_hash]
            if is_cache_enabled():
                self.register_buffer(padded_indices_buffer_name, padded_indices)
                self.register_buffer(
                    padded_normalized_distances_buffer_name, padded_normalized_distances
                )
                self.register_buffer(
                    padded_normalized_directions_buffer_name,
                    padded_normalized_directions,
                )
                self.register_buffer(mask_buffer_name, mask)
        elif (
            input_output_hash not in self.collection_matrix_cache
            and padded_indices_buffer_entry is None
        ):
            # Both are missing, so compute and cache the result
            (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            ) = self._block_collection_matrix(
                output_vectors, input_vectors, self.block_size
            )

            self.collection_matrix_cache[input_output_hash] = (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            )
            if is_cache_enabled():
                self.register_buffer(padded_indices_buffer_name, padded_indices)
                self.register_buffer(
                    padded_normalized_distances_buffer_name, padded_normalized_distances
                )
                self.register_buffer(
                    padded_normalized_directions_buffer_name,
                    padded_normalized_directions,
                )
                self.register_buffer(mask_buffer_name, mask)
        else:
            (
                padded_indices,
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
            ) = (
                padded_indices_buffer_entry,
                padded_normalized_distances_buffer_entry,
                padded_normalized_directions_buffer_entry,
                mask_buffer_entry,
            )

        return (
            padded_indices.to(device=_device),
            padded_normalized_distances.to(device=_device),
            padded_normalized_directions.to(device=_device),
            mask.to(device=_device),
        )

    def _query_weights(
        self,
        padded_normalized_distances: torch.Tensor,
        padded_normalized_directions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query MLPs weights for the candidates based on the normalized distances.

        Args:
            padded_normalized_distances (torch.Tensor): Tensor of shape (num_output_pixels, num_max_candidates, 1).
            padded_normalized_directions (torch.Tensor): Tensor of shape (num_output_pixels, num_max_candidates, 1).
            mask (torch.Tensor): Boolean mask of shape (num_output_pixels, num_max_candidates) indicating valid candidates.

        Returns:
            torch.Tensor: Candidate weights of shape (num_output_pixels, num_max_candidates, in_channels, out_channels).
        """
        num_output_pixels, num_max_candidates, _ = padded_normalized_distances.shape

        candidate_weights = torch.zeros(
            num_output_pixels,
            num_max_candidates,
            self.in_channels * self.out_channels,
            device=self.device,
            dtype=self.dtype,
        )  # invalid candidates are 0

        # Use MLP to compute weights.
        weights = checkpoint(
            self.weighting_function,
            {
                "distance": padded_normalized_distances[
                    mask
                ],  # shape: (num_valid_distances, 1)
                "direction": padded_normalized_directions[
                    mask
                ],  # shape: (num_valid_directions, 1)
            },
            use_reentrant=False,
        )
        # Elementwise multiplication
        candidate_weights[mask] = torch.prod(
            torch.cat(list(weights.values()), dim=-1), dim=-1
        )
        candidate_weights = candidate_weights.view(
            num_output_pixels, num_max_candidates, self.in_channels, self.out_channels
        )

        # Normalize each row with number of candidates
        candidate_counts = mask.float().sum(dim=1)
        valid_rows = candidate_counts > 0
        if valid_rows.any():
            candidate_weights[valid_rows] /= candidate_counts[valid_rows].view(
                -1, 1, 1, 1
            )

        return candidate_weights

    def _block_query_weights(
        self,
        padded_normalized_distances: torch.Tensor,
        padded_normalized_directions: torch.Tensor,
        mask: torch.Tensor,
        block_size: int = 2048,
    ) -> torch.Tensor:
        num_output_pixels, num_max_candidates, _ = padded_normalized_distances.shape

        # strictly constraining intermediate tensor size by limiting the total number of elements
        # ! magic equation here, otherwise would have to define a separate query_weight_chunk_size param
        chunk_size = math.floor(
            block_size
            / (num_max_candidates * math.sqrt(self.in_channels * self.out_channels))
        )
        chunk_size = max(1, chunk_size)  # avoid division by 0

        distance_chunks = torch.split(padded_normalized_distances, chunk_size, dim=0)
        direction_chunks = torch.split(padded_normalized_directions, chunk_size, dim=0)
        mask_chunks = torch.split(mask, chunk_size, dim=0)

        candidate_weight_chunks: list[torch.Tensor] = []
        for distance_chunk, direction_chunk, mask_chunk in zip(
            distance_chunks, direction_chunks, mask_chunks
        ):
            candidate_weight_chunks.append(
                self._query_weights(distance_chunk, direction_chunk, mask_chunk)
            )

        candidate_weight = torch.cat(candidate_weight_chunks, dim=0)

        return candidate_weight

    def _get_candidate_weights(
        self,
        padded_normalized_distances: torch.Tensor,
        padded_normalized_directions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute candidate weights based on the normalized distances and directions using the MLP in training mode.
        Or use the cached weights in eval mode.

        Args:
            padded_normalized_distances (torch.Tensor): Tensor of shape (num_output_pixels, num_max_candidates). Value range in [0.0, 1.0]
            padded_normalized_directions (torch.Tensor): Tensor of shape (num_output_pixels, num_max_candidates). Value range in [-1.0, 1.0]
            mask (torch.Tensor): Boolean mask of shape (num_output_pixels, num_max_candidates) indicating valid candidates.

        Returns:
            torch.Tensor: Candidate weights of shape (num_output_pixels, num_max_candidates, in_channels, out_channels).
        """
        if self.training:
            candidate_weights = self._block_query_weights(
                padded_normalized_distances,
                padded_normalized_directions,
                mask,
                self.block_size,
            )
        else:
            buffer_name = f"candidate_weights_{array_hash(padded_normalized_distances)}_{module_weights_hash(self.weighting_function)}"
            candidate_weights = getattr(self, buffer_name, None)

            if candidate_weights is None:
                # clear old buffers
                old_buffer_prefix = (
                    f"candidate_weights_{array_hash(padded_normalized_distances)}_"
                )
                to_remove = [
                    name
                    for name, _ in self.named_buffers()
                    if name.startswith(old_buffer_prefix)
                ]
                for name in to_remove:
                    delattr(
                        self, name
                    )  # ! PyTorch developer, please provide formal API to unregister buffers

                # register new buffer
                candidate_weights = self._block_query_weights(
                    padded_normalized_distances,
                    padded_normalized_directions,
                    mask,
                    self.block_size,
                )
                self.register_buffer(buffer_name, candidate_weights)

        return candidate_weights.to(device=self.device, dtype=self.dtype)

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """
        Full forward pass of RippleConv:
            1. Obtain input positions (num_input_pixels, 3) and input features (batch_size, num_input_pixels, in_channels).
            2. Compute output positions (num_output_pixels, 3) using the location sampler.
            3. Compute the collection matrix:
                padded_indices, padded_normalized_distances, mask of shape (num_output_pixels, num_max_candidates).
            4. Gather candidate input features using padded_indices.
            5. For each candidate, compute candidate_distance_embedding via harmonic embedding and then candidate weights using MLP.
            6. Multiply candidate input features by candidate weights and sum over candidate dimension to get output features.
            7. Add bias and return a BatchSphericalImage.
        """
        _device, _dtype = self.device, self.dtype
        input_vector_dtype = batch_spherical_image.vector.dtype

        batch_size, num_input_pixels, num_input_channels = (
            batch_spherical_image.batch_value.shape
        )
        assert (
            num_input_channels == self.in_channels
        ), f"in_channels mismatch: input_spherical_image has {num_input_channels}, but expected {self.in_channels}"

        # 1. Input positions and features.
        input_vectors = self.latest_input_vector = batch_spherical_image.vector.view(
            -1, 3
        ).to(device=_device, dtype=_dtype)

        # 2. Compute output positions via location sampler.
        output_vectors = self.latest_output_vector = self._get_output_vector(
            input_vectors
        ).to(device=_device, dtype=_dtype)

        # 3. Compute the collection matrix.
        #    All in shape (num_output_pixels, num_max_candidates)
        (
            padded_indices,
            padded_normalized_distances,
            padded_normalized_directions,
            mask,
        ) = self.latest_collection_matrix = self._get_collection_matrix(
            input_vectors, output_vectors
        )
        num_output_pixels, num_max_candidates = padded_indices.shape

        # 4. Get candidate weights.
        #    shape: (num_output_pixels, num_max_candidates, in_channels, out_channels)
        candidate_weights = self._get_candidate_weights(
            padded_normalized_distances, padded_normalized_directions, mask
        )

        # 5. Gather candidate input features.
        # Create a batch index tensor of shape (batch_size, num_output_pixels, num_max_candidates).
        batch_index = (
            torch.arange(batch_size, device=_device)
            .view(batch_size, 1, 1)
            .expand(batch_size, num_output_pixels, num_max_candidates)
        )

        # Expand padded_indices along the batch dimension and use advanced indexing to gather candidate features.
        # shape (batch_size, num_output_pixels, num_max_candidates, in_channels).
        candidate_input_features = batch_spherical_image.batch_value[
            batch_index,
            padded_indices.unsqueeze(0).expand(
                batch_size, num_output_pixels, num_max_candidates
            ),
        ]

        # 6. Weight candidate values.
        # candidate_input_features: (batch_size, num_output_pixels, num_max_candidates, in_channels)
        # candidate_weights: (num_output_pixels, num_max_candidates, in_channels, out_channels)
        # For each candidate, multiply input feature (over in_channels) with corresponding candidate weight, then sum over candidate dimension and input channels.
        # output_features: (batch_size, num_output_pixels, out_channels)
        output_features = contract(
            "bmki,mkio->bmo",
            candidate_input_features,
            candidate_weights,
            memory_limit="max_input",
        )

        # # (batch_size, num_output_pixels, 1, num_max_candidates x in_channels) @ (num_output_pixels, num_max_candidates x in_channels, out_channels)
        # # = (batch_size, num_output_pixels, 1, out_channels)
        # output_features = (
        #     candidate_input_features.view(batch_size, num_output_pixels, 1, -1)
        #     @ candidate_weights.view(num_output_pixels, -1, self.out_channels)
        # ).squeeze(-2)

        # 7. Add bias.
        if self.bias is not None:
            output_features += self.bias.view(1, 1, -1)

        output_batch_spherical_image = BatchSphericalImage(
            batch_value=output_features,
            vector=output_vectors.to(dtype=input_vector_dtype),
        )

        return output_batch_spherical_image

    @torch.inference_mode()
    def report_metrics(self, return_metrics: bool = False) -> dict | None:
        """Report statistics on the latest collection matrix for SphericalConv.

        Args:
            return_metrics (bool, optional): If True, return a dict of metrics. Defaults to False.

        Returns:
            dict | None: Metrics dict when ``return_metrics`` is True, else None.
        """
        if getattr(self, "latest_collection_matrix", None) is None:
            import warnings

            warnings.warn(
                "No collection matrix found. Run forward() first.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # latest_collection_matrix is assumed to be a tuple:
        # (padded_indices, padded_normalized_distances, mask)
        (
            padded_indices,
            padded_normalized_distances,
            padded_normalized_directions,
            mask,
        ) = self.latest_collection_matrix

        num_output_pixels, num_max_candidates = mask.shape
        num_input_pixels = self.latest_input_vector.shape[0]  # total input pixels

        # Convert to sparse-like format: flatten and select valid indices.
        out_idx = torch.arange(
            num_output_pixels, device=padded_indices.device
        ).repeat_interleave(num_max_candidates)
        in_idx = padded_indices.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid_out_idx = out_idx[mask_flat]
        valid_in_idx = in_idx[mask_flat]
        indices = torch.stack((valid_out_idx, valid_in_idx))  # shape: (2, num_valid)

        # Compute input pixels per output pixel.
        input_per_output = torch.bincount(
            indices[0], minlength=num_output_pixels
        ).float()
        avg_input_per_output = input_per_output.mean().item()
        std_input_per_output = input_per_output.std(unbiased=False).item()
        max_input_per_output = int(input_per_output.max().item())
        min_input_per_output = int(input_per_output.min().item())

        # Input coverage: fraction of input pixels that were used.
        unique_inputs = torch.unique(indices[1]).numel()
        input_coverage = unique_inputs / num_input_pixels

        # Output coverage: fraction of output pixels that received at least one input.
        output_coverage = (input_per_output > 0).float().mean().item()

        # Compute per-input access counts.
        input_access_counts = torch.bincount(
            indices[1], minlength=num_input_pixels
        ).float()
        accessed_inputs = input_access_counts[input_access_counts > 0]
        avg_access_per_input = (
            accessed_inputs.mean().item() if accessed_inputs.numel() > 0 else 0.0
        )
        std_access_per_input = (
            accessed_inputs.std(unbiased=False).item()
            if accessed_inputs.numel() > 0
            else 0.0
        )
        max_access_per_input = int(
            accessed_inputs.max().item() if accessed_inputs.numel() > 0 else 0
        )
        min_access_per_input = int(
            accessed_inputs.min().item() if accessed_inputs.numel() > 0 else 0
        )

        # Compute statistics for normalized distances.
        valid_normalized_distances = padded_normalized_distances[
            mask
        ]  # only valid candidates
        if valid_normalized_distances.numel() > 0:
            avg_normalized_distance = valid_normalized_distances.mean().item()
            std_normalized_distance = valid_normalized_distances.std(
                unbiased=False
            ).item()
            max_normalized_distance = valid_normalized_distances.max().item()
            min_normalized_distance = valid_normalized_distances.min().item()
        else:
            avg_normalized_distance = std_normalized_distance = (
                max_normalized_distance
            ) = min_normalized_distance = 0.0

        # Compute statistics for directions.
        valid_normalized_directions = padded_normalized_directions[
            mask
        ]  # only valid candidates
        if valid_normalized_directions.numel() > 0:
            avg_normalized_direction = valid_normalized_directions.mean().item()
            std_normalized_direction = valid_normalized_directions.std(
                unbiased=False
            ).item()
            max_normalized_direction = valid_normalized_directions.max().item()
            min_normalized_direction = valid_normalized_directions.min().item()
        else:
            avg_normalized_direction = std_normalized_direction = (
                max_normalized_direction
            ) = min_normalized_direction = 0.0

        # Total number of queries to weighting function
        total_weighting_queries = mask.sum().item()

        print("Spherical CNN Metrics:")
        print(
            f"\tAvg/Std/Max/Min # input pixels per output pixel: {round(avg_input_per_output, 3)}/{round(std_input_per_output, 3)}/{max_input_per_output}/{min_input_per_output}"
        )
        print(
            f"\tAvg/Std/Max/Min # access per input pixel: {round(avg_access_per_input, 3)}/{round(std_access_per_input, 3)}/{max_access_per_input}/{min_access_per_input}"
        )
        print(
            f"\tAvg/Std/Max/Min normalized distance: {round(avg_normalized_distance, 3)}/{round(std_normalized_distance, 3)}/{round(max_normalized_distance, 3)}/{round(min_normalized_distance, 3)}"
        )
        print(
            f"\tAvg/Std/Max/Min normalized direction: {round(avg_normalized_direction, 3)}/{round(std_normalized_direction, 3)}/{round(max_normalized_direction, 3)}/{round(min_normalized_direction, 3)}"
        )
        print(
            f"\tInput/Output pixel coverage: {round(input_coverage * 100, 1)}%/{round(output_coverage * 100, 1)}%"
        )
        print(
            f"\tpadded_indices shape/size: {padded_indices.shape}/{round(array_size_MiB(padded_indices), 3)} MiB"
        )
        print(f"\tmask shape/size: {mask.shape}/{round(array_size_MiB(mask), 3)} MiB")
        print(
            f"\tTotal number of queries to weighting function: {total_weighting_queries}"
        )

        if return_metrics:
            return {
                "avg_input_per_output": avg_input_per_output,
                "std_input_per_output": std_input_per_output,
                "max_input_per_output": max_input_per_output,
                "min_input_per_output": min_input_per_output,
                "avg_access_per_input": avg_access_per_input,
                "std_access_per_input": std_access_per_input,
                "max_access_per_input": max_access_per_input,
                "min_access_per_input": min_access_per_input,
                "avg_normalized_distance": avg_normalized_distance,
                "std_normalized_distance": std_normalized_distance,
                "max_normalized_distance": max_normalized_distance,
                "min_normalized_distance": min_normalized_distance,
                "avg_normalized_direction": avg_normalized_direction,
                "std_normalized_direction": std_normalized_direction,
                "max_normalized_direction": max_normalized_direction,
                "min_normalized_direction": min_normalized_direction,
                "input_coverage": input_coverage,
                "output_coverage": output_coverage,
                "total_weighting_queries": total_weighting_queries,
            }
        else:
            return None

    @torch.inference_mode()
    def visualize_kernel(
        self,
        visualize_input_location: bool = True,
        visualize_output_location: bool = True,
        num_display_kernel: int = 2,
        circle_spacing: float = math.pi / 100000,
        wave_height: float = 0.005,
        colormap: str = "hsv",
    ) -> SphericalImage:
        """
        Visualize the kernel weights of SphericalConv.

        Args:
            visualize_input_location (bool, optional): whether to see the latest input location. Defaults to True.
            visualize_output_location (bool, optional): whether to see the latest output location. Defaults to True.
            num_display_kernel (int, optional): number of kernel centers to visualize. Defaults to 2.
            circle_spacing (float, optional): spacing between sample points on circles. Defaults to math.pi / 100000.
            wave_height (float, optional): height of wave effect. Defaults to 0.005.
            colormap (str, optional): colormap to use for visualization. Defaults to "hsv".

        Returns:
            SphericalImage: A visualizable spherical image containing the kernel sample circles.
        """
        assert num_display_kernel > 0, "Must visualize at least one kernel"
        _device, _dtype = self.device, self.dtype

        # gather radii
        num_sample_points = max(3, int(self.radius / circle_spacing))
        # shape: (num_sample_points,)
        r = np.linspace(0, self.radius, num_sample_points, dtype=np.float32)[1:]

        # Get kernel centers (output positions). They should be computed during forward.
        if hasattr(self, "latest_output_vector") and isinstance(
            self.latest_output_vector, torch.Tensor
        ):
            kernel_centers = to_numpy(copy_or_clone(self.latest_output_vector))
            kernel_centers = ripple_sort(
                kernel_centers, np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
        else:
            # Fallback: choose a default center.
            kernel_centers = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        num_display_kernel = min(num_display_kernel, kernel_centers.shape[0])

        kernel_center_spherical_image = SphericalImage(
            value=colorize_locations(kernel_centers), vector=kernel_centers
        )

        # Select the first num_display_kernel centers.
        selected_centers = kernel_centers[:num_display_kernel]

        # sample circles around the selected centers
        # circle_points shape: (num_display_kernel, total_points, 3)
        circle_points = sample_circles(selected_centers, r, circle_spacing)

        # Get weight of one kernel
        # Get normalized_distances for the first num_display_kernel center.
        dot = circle_points[0] @ selected_centers[0, :].T  # shape: (total_points, 1)
        geodesic_distances = torch.acos(
            torch.clamp(to_torch(dot, device=_device, dtype=_dtype), -1.0, 1.0)
        )  # shape: (total_points, 1)
        normalized_distances = geodesic_distances / self.radius

        # Get directions
        input_points = to_torch(circle_points[0])
        reference_points = to_torch(selected_centers[0, :]).expand(
            input_points.shape[0], 3
        )
        normalized_directions = (
            relative_direction(reference_points, input_points) / torch.pi
        )

        # Query the MLP for candidate weights for these normalized distances.
        with torch.no_grad():
            normalized_distances = normalized_distances.to(
                device=_device, dtype=_dtype
            ).view(-1, 1, 1)
            normalized_directions = normalized_directions.to(
                device=_device, dtype=_dtype
            ).view(-1, 1, 1)
            total_points = normalized_distances.shape[0]
            candidate_weights = self._block_query_weights(
                normalized_distances,
                normalized_directions,
                torch.ones(
                    total_points,
                    1,
                    dtype=torch.bool,
                    device=_device,
                ),
                self.block_size,
            ).view(total_points, -1)

        candidate_weights_np = to_numpy(candidate_weights)
        # Compute variation for each channel
        variations = candidate_weights_np.max(axis=0) - candidate_weights_np.min(axis=0)
        # Select the channel index with maximum variation
        visual_weight = candidate_weights_np[
            :, int(np.argmax(variations))
        ]  # shape: (total_points,)

        # Normalize the visual_weight to [0, 1]
        visual_weight = (visual_weight - visual_weight.min()) / (
            visual_weight.max() - visual_weight.min()
        )

        # Generate color
        colors = cm.get_cmap(colormap)(visual_weight)[..., :3] * 255

        colors = colors[None, ...].repeat(
            num_display_kernel, axis=0
        )  # shape: (num_display_kernel, total_points, 3)

        # Elongate vectors
        elongation = 1.0 + wave_height * visual_weight.reshape(
            1, -1, 1
        )  # shape: (1, total_points, 1)
        circle_points *= elongation

        ripple_spherical_image = SphericalImage(value=colors, vector=circle_points)

        # Overlay input and output locations if desired.
        output_spherical_image = ripple_spherical_image
        if visualize_input_location:
            assert hasattr(
                self, "latest_input_vector"
            ), "Must run forward at least once to visualize kernel with input vector location"
            input_locations = to_numpy(copy_or_clone(self.latest_input_vector))
            input_colors = colorize_locations(
                input_locations,
                colormap="spring",
            )  # use a different colormap
            input_spherical_image = SphericalImage(
                value=input_colors, vector=input_locations
            )
            output_spherical_image += input_spherical_image
        if visualize_output_location:
            output_spherical_image += kernel_center_spherical_image

        return output_spherical_image


register_cache("SphericalConv.output_vector_cache", SphericalConv.output_vector_cache)
register_cache(
    "SphericalConv.collection_matrix_cache", SphericalConv.collection_matrix_cache
)
