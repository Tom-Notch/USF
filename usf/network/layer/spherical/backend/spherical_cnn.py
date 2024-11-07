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
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cachetools import LFUCache
from matplotlib import cm
from opt_einsum import contract
from torch.utils.checkpoint import checkpoint
from torch_scatter import scatter

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
                distance_config["value_min"] = 0.0
                distance_config["value_max"] = self.radius
            elif function_type == "continuous":
                # inject support-adapted base frequency for cosine distance embedding
                distance_embedding = distance_config.get("embedding")
                if (
                    distance_embedding is not None
                    and distance_embedding.get("type") == "cosine"
                ):
                    distance_embedding["basis"] = float(
                        math.floor(math.pi / self.radius)
                    )

        direction_config = weighting_function_config.get("direction", None)
        if direction_config is not None:
            function_type = direction_config.get("function", None)
            if function_type == "discrete":
                direction_config["value_min"] = -torch.pi
                direction_config["value_max"] = torch.pi

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
        input_vectors: torch.Tensor,
        output_vectors: torch.Tensor,
        block_size: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the collection matrix.

        For each output vector (of shape (num_output_pixels, 3)) and each input vector
        (of shape (num_input_pixels, 3)), if the dot product is above the cosine_threshold
        (i.e. within the influence range defined by radius), record the input index and
        compute the geodesic distance.

        Returns 2 tensors:
            - indices: edge list, shape (2, E), first row is input index, second output
            - relative_quantities: shape (2, E), first row is geodesic distance in [0, radius], second is direction in [-pi, pi]
        """
        assert (
            output_vectors.shape[-1] == input_vectors.shape[-1]
        ), "Last dimension must match"
        _device, _dtype = self.device, self.dtype

        # Unify device and dtype
        output_vectors = output_vectors.to(device=_device, dtype=_dtype)
        input_vectors = input_vectors.to(device=_device, dtype=_dtype)

        output_vectors.shape[0]
        input_vectors.shape[0]

        cosine_threshold = torch.cos(
            torch.tensor(self.radius, device=_device, dtype=_dtype)
        )

        membership_indices_list: list[torch.Tensor] = []
        relative_quantities_list: list[torch.Tensor] = []

        # Chunk
        output_chunks = torch.split(output_vectors, block_size, dim=0)
        input_chunks = torch.split(input_vectors, block_size, dim=0)

        output_offset = 0
        for output_chunk in output_chunks:
            input_offset = 0
            for input_chunk in input_chunks:
                # shape: (chunk_output_pixels, chunk_input_pixels)
                dot_block = torch.clamp(input_chunk @ output_chunk.T, -1.0, 1.0)
                valid_mask_block = dot_block >= cosine_threshold
                if valid_mask_block.sum() > 0:
                    # shape: (chunk_output_pixels, chunk_input_pixels)
                    distance_block = torch.acos(dot_block)
                    # shape: (num_valid, 2)
                    valid_indices = torch.nonzero(valid_mask_block)

                    valid_input_vector = input_chunk[valid_indices[:, 0]]
                    valid_output_vector = output_chunk[valid_indices[:, 1]]
                    relative_distances = distance_block[valid_mask_block].unsqueeze(
                        0
                    )  # (1, num_valid)
                    relative_directions = relative_direction(
                        valid_output_vector, valid_input_vector
                    ).unsqueeze(
                        0
                    )  # (1, num_valid)
                    relative_quantities = torch.vstack(
                        (relative_distances, relative_directions)
                    )  # (2, num_valid)
                    relative_quantities_list.append(relative_quantities)

                    valid_indices[:, 0] += input_offset
                    valid_indices[:, 1] += output_offset
                    membership_indices_list.append(valid_indices.T)  # (2, num_valid)

                input_offset += input_chunk.shape[0]

            output_offset += output_chunk.shape[0]

        indices = torch.hstack(membership_indices_list)
        quantities = torch.hstack(relative_quantities_list)

        return indices, quantities

    def _get_collection_matrix(
        self, input_vectors: torch.Tensor, output_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the cached collection matrix if available; otherwise computes it.

        The collection matrix consists of tensors:
            - indices: edge list, shape (2, E), first row is input index, second output
            - relative_quantities: shape (2, E), first row is geodesic distance in [0, radius], second is direction in [-pi, pi]

        These are computed based on input_vectors and output_vectors.
        """
        _device = self.device

        input_output_hash = (
            array_hash(torch.vstack((input_vectors, output_vectors)))
            + "_"
            + str(self.radius).replace(".", "dot")
        )
        indices_buffer_name = f"indices_{input_output_hash}"
        quantities_buffer_name = f"quantities_{input_output_hash}"

        # Get buffers (if they exist)
        indices_buffer_entry = getattr(self, indices_buffer_name, None)
        quantities_buffer_entry = getattr(self, quantities_buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_output_hash not in self.collection_matrix_cache
            and indices_buffer_entry is not None
        ):
            (
                indices,
                quantities,
            ) = (
                indices_buffer_entry,
                quantities_buffer_entry,
            )
            self.collection_matrix_cache[input_output_hash] = (
                indices,
                quantities,
            )
        elif (
            input_output_hash in self.collection_matrix_cache
            and indices_buffer_entry is None
        ):
            (
                indices,
                quantities,
            ) = self.collection_matrix_cache[input_output_hash]
            if is_cache_enabled():
                self.register_buffer(indices_buffer_name, indices)
                self.register_buffer(quantities_buffer_name, quantities)
        elif (
            input_output_hash not in self.collection_matrix_cache
            and indices_buffer_entry is None
        ):
            # Both are missing, so compute and cache the result
            (
                indices,
                quantities,
            ) = self._block_collection_matrix(
                input_vectors, output_vectors, self.block_size
            )

            self.collection_matrix_cache[input_output_hash] = indices, quantities
            if is_cache_enabled():
                self.register_buffer(indices_buffer_name, indices)
                self.register_buffer(quantities_buffer_name, quantities)
        else:
            (
                indices,
                quantities,
            ) = (
                indices_buffer_entry,
                quantities_buffer_entry,
            )

        return (
            indices.to(device=_device),
            quantities.to(device=_device),
        )

    def _query_weights(self, quantities: torch.Tensor) -> torch.Tensor:
        """
        Query MLPs weights for the candidates based on distances and directions.

        Args:
            quantities (torch.Tensor): Tensor of shape (2, E), first row is distance, second direction

        Returns:
            torch.Tensor: edge weights of shape (E, in_channels, out_channels).
        """
        # Use MLP to compute weights.
        weights = checkpoint(
            self.weighting_function,
            {
                "distance": quantities[0].unsqueeze(1),  # shape: (num_edges, 1)
                "direction": quantities[1].unsqueeze(1),  # shape: (num_edges, 1)
            },
            use_reentrant=False,
        )
        # Element-wise multiplication
        edge_weights = torch.prod(torch.cat(list(weights.values()), dim=-1), dim=-1)
        edge_weights = edge_weights.view(-1, self.in_channels, self.out_channels)

        return edge_weights

    def _block_query_weights(
        self,
        quantities: torch.Tensor,
        block_size: int = 2048,
    ) -> torch.Tensor:
        quantities.shape[1]

        # quantities shape (2, num_edges)
        quantity_chunks = torch.split(quantities, block_size, dim=1)

        candidate_weight = torch.cat(
            [self._query_weights(quantity_chunk) for quantity_chunk in quantity_chunks],
            dim=0,
        )

        return candidate_weight

    def _get_candidate_weights(self, quantities: torch.Tensor) -> torch.Tensor:
        """
        Compute candidate weights based on distances and directions using the MLP in training mode.
        Or use the cached weights in eval mode.

        Args:
            quantities (torch.Tensor): Tensor of shape (2, num_edges), first row is geodesic distance, second is direction.

        Returns:
            torch.Tensor: Candidate weights of shape (num_edges, in_channels, out_channels).
        """
        if self.training:
            candidate_weights = self._block_query_weights(
                quantities,
                self.block_size,
            )
        else:
            buffer_name = f"candidate_weights_{array_hash(quantities)}_{module_weights_hash(self.weighting_function)}"
            candidate_weights = getattr(self, buffer_name, None)

            if candidate_weights is None:
                # clear old buffers
                old_buffer_prefix = f"candidate_weights_{array_hash(quantities)}_"
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
                    quantities,
                    self.block_size,
                )
                self.register_buffer(buffer_name, candidate_weights)

        return candidate_weights.to(device=self.device, dtype=self.dtype)

    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """
        Full forward pass of SphericalConv:
            1. Obtain input positions (num_input_pixels, 3) and input features (batch_size, num_input_pixels, in_channels).
            2. Compute output positions (num_output_pixels, 3) using the location sampler.
            3. Compute the collection matrix:
                indices, quantities, mask of shape (num_output_pixels, num_max_candidates).
            4. Gather candidate input features using indices.
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
        #    All in shape (2, num_edges)
        indices, relative_quantities = self.latest_collection_matrix = (
            self._get_collection_matrix(input_vectors, output_vectors)
        )

        # 4. Get edge weights.
        #    shape: (num_edges, in_channels, out_channels)
        edge_weights = self._get_candidate_weights(relative_quantities)

        # 5. Gather edge features
        #    Expand indices along the batch dimension and use advanced indexing to gather candidate features.
        #    shape (B, num_edges, in_channels).
        edge_features = batch_spherical_image.batch_value[:, indices[0]]

        # 6. Weight edge values.
        # edge_features: (batch_size, num_edges, in_channels)
        # edge_weights: (num_edges, in_channels, out_channels)
        # For each edge, left-multiply feature with corresponding edge weight matrix.
        # weighted_edge_batch_value: (batch_size, num_edges, out_channels)
        weighted_edge_features = contract(
            "bei,eio->beo",
            edge_features,
            edge_weights,
            memory_limit="max_input",
        )

        # 7. Scatter into output vector locations/values
        #    NOTE: scatter with reduce="mean" uses atomicAdd on CUDA, which is
        #    non-deterministic in floating-point summation order. Identical batch
        #    elements may produce slightly different outputs. segment_csr is a
        #    deterministic alternative but ~3x slower at realistic edge counts
        #    (E~920K). We keep scatter for performance.
        num_output_vectors = output_vectors.shape[0]
        output_features = torch.zeros(
            (batch_size, num_output_vectors, self.out_channels),
            device=weighted_edge_features.device,
            dtype=weighted_edge_features.dtype,
        )
        scatter(
            src=weighted_edge_features,
            index=indices[1],
            dim=-2,
            out=output_features,
            reduce="mean",
        )

        # 8. Add bias.
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
            warnings.warn(
                "No collection matrix found. Run forward() first.",
                UserWarning,
                stacklevel=2,
            )
            return None

        indices, quantities = self.latest_collection_matrix
        in_idx = indices[0]
        out_idx = indices[1]
        distances = quantities[0]  # geodesic distance in [0, radius]
        directions = quantities[1]  # relative direction in [-pi, pi]

        num_input_pixels = self.latest_input_vector.shape[0]
        num_output_pixels = self.latest_output_vector.shape[0]

        # Inputs per output
        input_per_output = torch.bincount(out_idx, minlength=num_output_pixels).float()
        avg_input_per_output = input_per_output.mean().item()
        std_input_per_output = input_per_output.std(unbiased=False).item()
        max_input_per_output = int(input_per_output.max().item())
        min_input_per_output = int(input_per_output.min().item())

        # Access per input
        input_access_counts = torch.bincount(in_idx, minlength=num_input_pixels).float()
        accessed = input_access_counts[input_access_counts > 0]
        avg_access_per_input = accessed.mean().item()
        std_access_per_input = accessed.std(unbiased=False).item()
        max_access_per_input = int(accessed.max().item())
        min_access_per_input = int(accessed.min().item())

        # Coverages
        input_coverage = in_idx.unique().numel() / num_input_pixels
        output_coverage = (input_per_output > 0).float().mean().item()

        avg_distance = distances.mean().item()
        std_distance = distances.std(unbiased=False).item()
        max_distance = distances.max().item()
        min_distance = distances.min().item()

        avg_direction = directions.mean().item()
        std_direction = directions.std(unbiased=False).item()
        max_direction = directions.max().item()
        min_direction = directions.min().item()

        # Total number of queries to weighting function
        total_weighting_queries = indices.shape[1]

        print("Spherical CNN Metrics:")
        print(
            f"\tAvg/Std/Max/Min # input pixels per output pixel: {round(avg_input_per_output, 3)}/{round(std_input_per_output, 3)}/{max_input_per_output}/{min_input_per_output}"
        )
        print(
            f"\tAvg/Std/Max/Min # access per input pixel: {round(avg_access_per_input, 3)}/{round(std_access_per_input, 3)}/{max_access_per_input}/{min_access_per_input}"
        )
        print(
            f"\tAvg/Std/Max/Min distance: {round(avg_distance, 3)}/{round(std_distance, 3)}/{round(max_distance, 3)}/{round(min_distance, 3)}"
        )
        print(
            f"\tAvg/Std/Max/Min direction: {round(avg_direction, 3)}/{round(std_direction, 3)}/{round(max_direction, 3)}/{round(min_direction, 3)}"
        )
        print(
            f"\tInput/Output pixel coverage: {round(input_coverage * 100, 1)}%/{round(output_coverage * 100, 1)}%"
        )
        print(
            f"\tindices shape/size: {indices.shape}/{round(array_size_MiB(indices), 3)} MiB"
        )
        print(
            f"\tquantities shape/size: {quantities.shape}/{round(array_size_MiB(quantities), 3)} MiB"
        )
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
                "avg_distance": avg_distance,
                "std_distance": std_distance,
                "max_distance": max_distance,
                "min_distance": min_distance,
                "avg_direction": avg_direction,
                "std_direction": std_direction,
                "max_direction": max_direction,
                "min_direction": min_direction,
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
            kernel_centers = np.array([[1, 0, 0]], dtype=np.float32)

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
        dot = circle_points[0] @ selected_centers[0, :].T  # shape: (total_points, 1)
        distances = torch.acos(
            torch.clamp(to_torch(dot, device=_device, dtype=_dtype), -1.0, 1.0)
        )  # shape: (total_points, 1)

        # Get directions
        input_points = to_torch(circle_points[0], dtype=_dtype)
        reference_points = to_torch(selected_centers[0, :], dtype=_dtype).expand(
            input_points.shape[0], 3
        )
        directions = relative_direction(reference_points, input_points)

        with torch.no_grad():
            distances = distances.to(device=_device, dtype=_dtype).view(1, -1)
            directions = directions.to(device=_device, dtype=_dtype).view(1, -1)
            quantities = torch.vstack((distances, directions))
            total_points = quantities.shape[-1]
            candidate_weights = self._block_query_weights(
                quantities, self.block_size
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
                input_locations, colormap="spring"
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
