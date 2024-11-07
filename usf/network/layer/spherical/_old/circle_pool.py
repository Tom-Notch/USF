#!/usr/bin/env python3
#
# Created on Fri Feb 07 2025 17:27:03
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from cachetools import LFUCache

from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.spherical import colorize_locations, ripple_sort
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import array_hash, array_size_MiB, copy_or_clone, to_numpy
from usf.visualization.spherical_layer import sample_circles


class CirclePool(nn.Module):
    """Geodesic circle pooling on S2 (legacy implementation).

    Collects input pixels within a geodesic radius around each output pixel
    and reduces via max, min, or mean pooling. Collection matrices are
    cached at the class level.
    """

    # class level cache shared among instances, need to guarantee uniqueness
    output_vector_cache = LFUCache(maxsize=500)
    collection_matrix_cache = LFUCache(maxsize=500)

    @staticmethod
    # @torch.compile
    def pool_max(candidate_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Max pool over candidates, masking invalid entries to -inf.

        Args:
            candidate_values (torch.Tensor): shape (B, N_out, K, C).
            mask (torch.Tensor): boolean shape (B, N_out, K, 1).

        Returns:
            torch.Tensor: shape (B, N_out, C).
        """
        candidate_values = torch.where(
            mask, candidate_values, torch.full_like(candidate_values, -float("inf"))
        )
        pooled, _ = candidate_values.max(dim=2)

        # For each image in the batch, find the per-image safe fallback:
        # That is, for each row in pooled, find the minimum value that is not -inf.
        valid_mask = pooled != -float("inf")

        # Replace -inf entries with +inf so they don't interfere with the min operation
        # when we are trying to find the safe fallback
        pooled_for_min = torch.where(
            valid_mask, pooled, torch.full_like(pooled, float("inf"))
        )
        fallback, _ = pooled_for_min.min(dim=1, keepdim=True)  # shape: (B, 1)

        # Check if any image has no valid entries (fallback == +infinity).
        # If so, warn once and replace those fallback values with 0.
        if (fallback == float("inf")).nonzero().size(0) > 0:
            warnings.warn(
                "Some images have no valid input vector that falls into any output vector's circle. Replacing with zero.",
                UserWarning,
            )
            fallback = torch.where(
                fallback == float("inf"), torch.zeros_like(fallback), fallback
            )

        # Now replace all -inf entries in pooled with the per-image fallback value
        pooled = torch.where(valid_mask, pooled, fallback)

        return pooled

    @staticmethod
    # @torch.compile
    def pool_min(candidate_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Min pool over candidates, masking invalid entries to +inf.

        Args:
            candidate_values (torch.Tensor): shape (B, N_out, K, C).
            mask (torch.Tensor): boolean shape (B, N_out, K, 1).

        Returns:
            torch.Tensor: shape (B, N_out, C).
        """
        candidate_values = torch.where(
            mask, candidate_values, torch.full_like(candidate_values, float("inf"))
        )
        # shape (batch_size, num_out_pixels)
        pooled, _ = candidate_values.min(dim=2)

        # For each image in the batch, find the per-image safe fallback:
        # That is, for each row in pooled, find the max value that is not +inf.
        valid_mask = pooled != float("inf")

        # Replace inf entries with -inf so they don't interfere with the max operation
        # when we are trying to find the safe fallback
        pooled_for_min = torch.where(
            valid_mask, pooled, torch.full_like(pooled, -float("inf"))
        )
        fallback, _ = pooled_for_min.max(dim=1, keepdim=True)  # shape: (B, 1)

        # Check if any image has no valid entries (fallback == -infinity).
        # If so, warn once and replace those fallback values with 0.
        if (fallback == -float("inf")).nonzero().size(0) > 0:
            warnings.warn(
                "Some images have no valid input vector that falls into any output vector's circle. Replacing with zero.",
                UserWarning,
            )
            fallback = torch.where(
                fallback == -float("inf"), torch.zeros_like(fallback), fallback
            )

        # Now replace all +inf entries in pooled with the per-image fallback value
        pooled = torch.where(valid_mask, pooled, fallback)

        return pooled

    @staticmethod
    # @torch.compile
    def pool_mean(candidate_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean pool over candidates, masking invalid entries to 0.

        Args:
            candidate_values (torch.Tensor): shape (B, N_out, K, C).
            mask (torch.Tensor): boolean shape (B, N_out, K, 1).

        Returns:
            torch.Tensor: shape (B, N_out, C).
        """
        # Step 1: Replace invalid candidate values with 0.
        # candidate_values_valid: shape (batch_size, num_out_pixels, num_candidates)
        candidate_values_valid = torch.where(
            mask, candidate_values, torch.zeros_like(candidate_values)
        )
        # Step 2: Sum candidate values over the candidate dimension.
        # sum_values: shape (batch_size, num_out_pixels)
        sum_values = candidate_values_valid.sum(dim=2)

        # Step 3: Count the number of valid candidates in each pooling region.
        # valid_counts: shape (batch_size, num_out_pixels)
        valid_counts = mask.sum(dim=2)  # could have 0s

        # Step 4: Compute the mean for each pooling region.
        # For regions with no valid candidate, use a temporary denominator of 1 to avoid division by zero.
        # pooled: shape (batch_size, num_out_pixels)
        pooled = sum_values / torch.clamp(valid_counts, min=1)

        # Step 5: Compute a per-image global fallback (the global mean) using all candidate positions.
        # First, flatten the candidate_values and mask.
        # candidate_values_flat: shape (batch_size, num_out_pixels * num_candidates)
        candidate_values_flat = candidate_values.view(candidate_values.shape[0], -1)
        # mask_flat: shape (batch_size, num_out_pixels * num_candidates)
        mask_flat = mask.view(mask.shape[0], -1)
        # Compute the sum over all valid candidate values for each image.
        # global_sum: shape (batch_size, 1)
        global_sum = torch.where(
            mask_flat, candidate_values_flat, torch.zeros_like(candidate_values_flat)
        ).sum(dim=1, keepdim=True)
        # Compute the count of valid candidate values for each image.
        # global_count: shape (batch_size, 1)
        global_count = mask_flat.sum(dim=1, keepdim=True)  # could have 0s
        # Compute the global mean for each image.
        # global_mean: shape (batch_size, 1)
        global_mean = global_sum / torch.clamp(global_count, min=1)

        # Step 6: Identify pooling regions with no valid candidate entries.
        # empty_region: shape (batch_size, num_out_pixels)
        empty_region = valid_counts == 0

        # Step 7: Warn if an entire image has no valid entries.
        if (global_count == 0).nonzero().size(0) > 0:
            warnings.warn(
                "Some images have no valid entries in any pooling window for mean pooling. Replacing with zero.",
                UserWarning,
            )
            global_mean = torch.where(
                global_count == 0, torch.zeros_like(global_mean), global_mean
            )

        # Step 8: Replace pooled values in regions with no valid candidate with the global mean for that image.
        # Expand global_mean to shape (batch_size, num_out_pixels).
        fallback_expanded = global_mean.expand_as(pooled)
        pooled = torch.where(empty_region, fallback_expanded, pooled)

        # Final pooled: shape (batch_size, num_out_pixels)
        return pooled

    _POOL_TYPE = {
        "max": pool_max.__func__,
        "min": pool_min.__func__,
        "mean": pool_mean.__func__,
    }

    def __init__(
        self,
        pool_type: str,
        radius: float,
        identical_output_vector: bool = False,
        resolution_factor: float = 1.0,
        location_sampler: str = "icosahedron",
        reject_oo_fov_vector: bool = True,
        block_size: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """circle/radial pooling on spherical signals (S2)

        Args:
            pool_type (str): "max", "min", or "mean"
            radius (float): in radians, geodesic radius of the pooling circle, analogous to kernel size
            identical_output_vector (bool, optional): whether output_vector is forcefully set to just input_vector
            resolution_factor (float, optional): factor to in/decrease the resolution of the input SphericalImage
                                       # output_pixels \approx resolution_factor * # input_pixels
            location_sampler (str, optional): which LocationSample to use for novel pixel location.
                                     Defaults to None which corresponds to icosahedron.
            reject_oo_fov_vector (bool, optional): whether to use panorama mode for location sampler, turning on will disable out-of-view vector rejection. Defaults to False.
            block_size (int, optional): the size of a block for large pairwise dot, each block is chunked into block_size x block_size. Can enlarge depending on how big your (V)RAM is. Defaults to 2048.
            device (torch.device | None, optional): torch device. Defaults to None.
            dtype (torch.dtype | None, optional): torch data type. Defaults to None.
        """
        super().__init__()

        assert (
            pool_type in self._POOL_TYPE.keys()
        ), f"pool_type {pool_type} must be one of {self._POOL_TYPE.keys()}"
        assert 0.0 < radius < torch.pi, f"radius {radius} must be in (0, π)"
        assert (
            resolution_factor > 0.0
        ), f"resolution_factor {resolution_factor} must be > 0.0"
        assert block_size > 0, "block_size must be positive"

        self.pool_type = pool_type
        self.pool_function = self._POOL_TYPE[pool_type]

        self.radius = radius

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
        self.device, self.dtype = device, dtype
        self.to(device=device, dtype=dtype)

    def extra_repr(self) -> str:
        """For printing Module information

        Returns:
            str: args and kwargs of the Module
        """
        extra_repr = f"type={self.pool_type}" f", radius={self.radius:.5f}"

        if isinstance(getattr(self, "override_output_vector", None), torch.Tensor):
            extra_repr += ", override_output_vector=True"
        elif self.identical_output_vector:
            extra_repr += ", identical_output_vector=True"

        extra_repr += f", block_size={self.block_size}"

        return extra_repr

    def set_output_vector(self, output_vector: torch.Tensor | None) -> CirclePool:
        """Set a fixed output vector that overrides any existing rule

        Args:
            output_vector (torch.Tensor | None): in shape (N, 3), each row is a cartesian coordinate

        Touches:
            self.override_output_vector: internal record

        Returns:
            self
        """
        if output_vector is not None:
            assert (
                output_vector.shape[-1] == 3
            ), f"Output_vector has last dimension = {output_vector.shape[-1]} which is not Cartesian"

            self.register_buffer(
                "override_output_vector",
                output_vector.detach().clone(),
            )

            if getattr(self, "location_sampler", None) is not None:
                del self.location_sampler
        elif getattr(self, "override_output_vector", None) is not None:
            del self.override_output_vector

            if getattr(self, "location_sampler", None) is None:
                self.location_sampler = LocationSampler(
                    config={
                        "location_sampler": self.location_sampler_type,
                        "resolution_factor": self.resolution_factor,
                        "reject_oo_fov_vector": self.reject_oo_fov_vector,
                    }
                )

        return self

    @torch.no_grad()
    def _block_collection_matrix(
        self, out_vector: torch.Tensor, in_vector: torch.Tensor, block_size: int = 2048
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blocked collection matrix creation for circle pooling.

        Args:
            out_vector (torch.Tensor): shape (M, d)
            in_vector (torch.Tensor): shape (N, d)
            block_size (int, optional): Size of each block for chunking. Defaults to 2048.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - padded_indices: shape (num_out_pixels, max_pool_size)
                - mask: shape (num_out_pixels, max_pool_size)
        """
        assert out_vector.shape[-1] == in_vector.shape[-1], "Last dimension must match"

        # Localize device
        _device, _dtype = self.device, self.dtype

        # Define variables
        num_out_pixels, num_in_pixels = out_vector.shape[0], in_vector.shape[0]

        # Compute circle cutoff cosine similarity
        circle_cos = torch.cos(torch.tensor(self.radius, device=_device, dtype=_dtype))

        # Unify device and dtype
        out_vector = out_vector.to(device=_device, dtype=_dtype)
        in_vector = in_vector.to(device=_device, dtype=_dtype)

        # Chunk `out_vector`
        out_vector_chunks = torch.split(out_vector, block_size, dim=0)

        # Chunk `in_vector`
        in_vector_chunks = torch.split(in_vector, block_size, dim=0)

        # Track processed rows/cols
        num_processed_columns = 0
        circle_membership_index_blocks = []

        for in_vector_chunk in in_vector_chunks:
            num_processed_rows = 0

            for out_vector_chunk in out_vector_chunks:
                # Compute dot product
                # shape: (out_vector_chunk.shape[0], in_vector_chunk.shape[0])
                dot_block = out_vector_chunk @ in_vector_chunk.T

                # Determine membership (binary mask)
                # shape: (K, 2), each row represents [out_idx, in_idx] pairs
                circle_membership_index_block = torch.argwhere(dot_block >= circle_cos)

                # Shift row indices for out_vector chunk
                circle_membership_index_block[:, 0] += num_processed_rows
                # Shift column indices for in_vector chunk
                circle_membership_index_block[:, 1] += num_processed_columns

                # Store block
                circle_membership_index_blocks.append(circle_membership_index_block)
                num_processed_rows += out_vector_chunk.shape[0]

                del dot_block

            num_processed_columns += in_vector_chunk.shape[0]

        # Concatenate all blocks
        if len(circle_membership_index_blocks) > 0:
            circle_membership_index = torch.vstack(circle_membership_index_blocks)
        else:
            # If no valid indices were found, return empty tensors
            return (
                torch.zeros((num_out_pixels, 1), dtype=torch.int64, device=_device),
                torch.zeros((num_out_pixels, 1), dtype=torch.bool, device=_device),
            )

        # Sort by out_idx to maintain order
        _, order = torch.sort(circle_membership_index[:, 0])
        circle_membership_index = circle_membership_index[order]

        # Compute per-output pooling sizes
        unique_out, counts = torch.unique_consecutive(
            circle_membership_index[:, 0], return_counts=True
        )
        max_pool_size = int(counts.max().item()) if counts.numel() > 0 else 0

        # Initialize padded indices and mask
        padded_indices = torch.zeros(
            (num_out_pixels, max_pool_size), dtype=torch.int64, device=_device
        )
        mask = torch.zeros(
            (num_out_pixels, max_pool_size), dtype=torch.bool, device=_device
        )

        # Compute cumulative counts to segment indices
        cum_counts = torch.cat(
            (torch.tensor([0], device=_device), counts.cumsum(dim=0))
        )

        # Fill in padded indices and mask
        for i in range(unique_out.shape[0]):
            out_idx = unique_out[i].item()  # Output index
            start_idx = cum_counts[i].item()
            end_idx = cum_counts[i + 1].item()
            num_vals = end_idx - start_idx

            padded_indices[out_idx, :num_vals] = circle_membership_index[
                start_idx:end_idx, 1
            ]
            mask[out_idx, :num_vals] = True

        return padded_indices, mask

    # @torch.compile
    def _pool_batch_image(
        self,
        batch_value: torch.Tensor,
        padded_indices: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pools a batch of spherical image values using precomputed padded indices and a mask

        Args:
            batch_value (torch.Tensor): Tensor of shape (batch_size, num_input_pixels, num_channels)
            padded_indices (torch.Tensor): torch.int64 Tensor of shape (num_output_pixels, max_pool_size)
                where each row contains candidate input indices (invalid positions are padded with 0)
            mask (torch.Tensor): torch.bool Tensor of shape (num_output_pixels, max_pool_size)
                indicating valid entries (True for valid, False for padded)

        Returns:
            torch.Tensor: Pooled output of shape (batch_size, num_output_pixels, num_channels)
        """
        # Get dimensions with descriptive variable names
        batch_size, num_input_pixels, num_channels = batch_value.shape
        num_output_pixels, max_pool_size = padded_indices.shape

        # Create a batch index tensor of shape (batch_size, num_output_pixels, max_pool_size)
        batch_index = (
            torch.arange(batch_size, device=batch_value.device)
            .view(batch_size, 1, 1)
            .expand(batch_size, num_output_pixels, max_pool_size)
        )

        # Expand padded_indices from shape (num_output_pixels, max_pool_size) to
        # (batch_size, num_output_pixels, max_pool_size)
        expanded_padded_indices = padded_indices.unsqueeze(0).expand(
            batch_size, num_output_pixels, max_pool_size
        )

        # Use fancy indexing to gather candidate values
        # For each batch sample b, each output pixel o, and each candidate position p:
        # candidate_values[b, o, p, :] = batch_value[b, expanded_padded_indices[b, o, p], :]
        candidate_values = batch_value[batch_index, expanded_padded_indices]
        # candidate_values now has shape (batch_size, num_output_pixels, max_pool_size, num_channels)

        # Expand mask to match candidate_values
        # shape (batch_size, num_output_pixels, max_pool_size, num_channels)
        expanded_mask = (
            mask.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, num_output_pixels, max_pool_size, num_channels)
        )

        # Call the pre-configured pooling function
        # For example, for max pooling, pool_function will replace invalid entries with -∞ and take the max along dimension 2
        pooled_output = self.pool_function(candidate_values, expanded_mask)
        # pooled_output will have shape (batch_size, num_output_pixels, num_channels)

        return pooled_output

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

    def _get_collection_matrix(
        self, input_vector: torch.Tensor, output_vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # class level uniqueness guaranteed
        input_output_vector_hash = (
            array_hash(torch.vstack((input_vector, output_vector)))
            + "_"
            + str(self.radius).replace(".", "dot")  # buffer name doesn't allow '.'
        )
        padded_indices_buffer_name = f"padded_indices_{input_output_vector_hash}"
        mask_buffer_name = f"mask_{input_output_vector_hash}"

        # Get cache and buffer (if they exist)
        padded_indices_buffer_entry = getattr(self, padded_indices_buffer_name, None)
        mask_buffer_entry = getattr(self, mask_buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_output_vector_hash not in self.collection_matrix_cache
            and padded_indices_buffer_entry is not None
        ):
            padded_indices, mask = (
                padded_indices_buffer_entry,
                mask_buffer_entry,
            )
            self.collection_matrix_cache[input_output_vector_hash] = (
                padded_indices,
                mask,
            )
        elif (
            input_output_vector_hash in self.collection_matrix_cache
            and padded_indices_buffer_entry is None
        ):
            padded_indices, mask = self.collection_matrix_cache[
                input_output_vector_hash
            ]
            if is_cache_enabled():
                self.register_buffer(padded_indices_buffer_name, padded_indices)
                self.register_buffer(mask_buffer_name, mask)
        elif (
            input_output_vector_hash not in self.collection_matrix_cache
            and padded_indices_buffer_entry is None
        ):
            # Both are missing, so compute and cache the result
            padded_indices, mask = self._block_collection_matrix(
                output_vector, input_vector, self.block_size
            )

            self.collection_matrix_cache[input_output_vector_hash] = (
                padded_indices,
                mask,
            )
            if is_cache_enabled():
                self.register_buffer(padded_indices_buffer_name, padded_indices)
                self.register_buffer(mask_buffer_name, mask)
        else:
            padded_indices, mask = padded_indices_buffer_entry, mask_buffer_entry

        return padded_indices.to(device=self.device), mask.to(device=self.device)

    # can use @torch.compile on hardware with larger GPU memory
    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Main forward pass, has complicated caching & pooling function mechanism

        Args:
            batch_spherical_image (BatchSphericalImage): batch_value in format (batch_size, num_in_pixels, in_channels)

        Touches:
            self.collection_matrix_cache: for speedup when encountering the same input vector again
            self.output_vector_cache: for speedup when encountering the same input vector again
            self.latest_output_vector: for kernel visualization

        Returns:
            BatchSphericalImage: batch_value in format (batch_size, num_out_pixels, in_channels)
        """
        _device = self.device = batch_spherical_image.batch_value.device
        _dtype = self.dtype = batch_spherical_image.batch_value.dtype

        input_vector = batch_spherical_image.vector.view(-1, 3)
        self.latest_input_vector = input_vector.to(device=_device, dtype=_dtype)

        output_vector = self._get_output_vector(input_vector)
        self.latest_output_vector = output_vector.to(device=_device, dtype=_dtype)

        padded_indices, mask = self._get_collection_matrix(input_vector, output_vector)
        self.latest_collection_matrix = padded_indices, mask  # for metrics report

        batch_pooled_value = self._pool_batch_image(
            batch_spherical_image.batch_value, padded_indices, mask
        )

        return BatchSphericalImage(batch_value=batch_pooled_value, vector=output_vector)

    @torch.inference_mode()
    def report_metrics(self, return_metrics: bool = False) -> dict | None:
        """Report statistics on the latest pooling collection matrix.

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

        # Retrieve latest cached (padded_indices, mask)
        padded_indices, mask = self.latest_collection_matrix

        num_out_pixels, max_pool_size = mask.shape
        num_in_pixels = self.latest_input_vector.shape[0]  # Correct total input pixels

        # --- Convert to Sparse Format (Like .indices() of a Sparse Matrix) ---
        # Flatten and filter valid indices using mask
        out_idx = torch.arange(
            num_out_pixels, device=padded_indices.device
        ).repeat_interleave(max_pool_size)
        in_idx = padded_indices.reshape(-1)
        mask_flat = mask.reshape(-1)

        # Keep only valid (output, input) index pairs
        valid_out_idx = out_idx[mask_flat]
        valid_in_idx = in_idx[mask_flat]

        # Stack into a (2, num_valid_entries) tensor
        indices = torch.stack((valid_out_idx, valid_in_idx))

        # --- 1. Avg / Std / Max / Min of Input Pixels Per Output Pixel ---
        input_per_output = torch.bincount(indices[0], minlength=num_out_pixels).float()
        avg_input_per_output = input_per_output.mean().item()
        std_input_per_output = input_per_output.std(unbiased=False).item()
        max_input_per_output = int(input_per_output.max().item())
        min_input_per_output = int(input_per_output.min().item())

        # --- 2. Input Coverage (Fraction of Input Pixels Used At Least Once) ---
        unique_inputs = torch.unique(indices[1]).numel()
        input_coverage = unique_inputs / num_in_pixels

        # --- 3. Output Coverage (Fraction of Output Pixels Receiving Input Pixels) ---
        output_coverage = (input_per_output > 0).float().mean().item()

        # --- 4. Avg / Std / Max / Min of Pool Size Per Accessed Input Pixel ---
        input_access_counts = torch.bincount(
            indices[1], minlength=num_in_pixels
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
            accessed_inputs.max().item() if accessed_inputs.numel() > 0 else 0.0
        )
        min_access_per_input = int(
            accessed_inputs.min().item() if accessed_inputs.numel() > 0 else 0.0
        )

        # Report results (Rounding Only in Print)
        print("Circle Pooling Metrics:")
        print(
            f"\tAvg/Std/Max/Min # input pixels per output pixel: {round(avg_input_per_output, 3)}/{round(std_input_per_output, 3)}/{max_input_per_output}/{min_input_per_output}"
        )
        print(
            f"\tAvg/Std/Max/Min # access per input pixel: {round(avg_access_per_input, 3)}/{round(std_access_per_input, 3)}/{max_access_per_input}/{min_access_per_input}"
        )
        print(
            f"\tInput/Output pixel coverage: {round(input_coverage * 100, 1)}%/{round(output_coverage * 100, 1)}%"
        )
        print(
            f"\tpadded_indices shape/size: {padded_indices.shape}/{round(array_size_MiB(padded_indices), 3)} MiB"
        )
        print(f"\tmask shape/size: {mask.shape}/{round(array_size_MiB(mask), 3)} MiB")

        return (
            {
                "avg_input_per_output": avg_input_per_output,
                "std_input_per_output": std_input_per_output,
                "max_input_per_output": max_input_per_output,
                "min_input_per_output": min_input_per_output,
                "avg_access_per_input": avg_access_per_input,
                "std_access_per_input": std_access_per_input,
                "max_access_per_input": max_access_per_input,
                "min_access_per_input": min_access_per_input,
                "input_coverage": input_coverage,
                "output_coverage": output_coverage,
            }
            if return_metrics
            else None
        )

    @torch.inference_mode()
    def visualize_kernel(
        self,
        visualize_input_location: bool = True,
        visualize_output_location: bool = True,
        num_display_kernel: int = 10,
        circle_group_center: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32),
        circle_spacing: float = 2 * np.pi / 5000,
    ) -> SphericalImage:
        """visualize the output location and several circles

        Args:
            visualize_input_location (bool, optional): whether to see the latest input location. Defaults to True.
            visualize_output_location (bool, optional): whether to see the latest output location. Defaults to True.
            num_display_kernel (int, optional): number of circle center locations, e.g. the first 5. Defaults to 5.
            circle_group_center (np.ndarray, optional): in cartesian coordinate, the point where center location are around. Defaults to np.array([-1, 0, 0])
            circle_spacing (float, optional): spacing between 2 sample points on a circle. Defaults to 2 * np.pi / 5000.

        Returns:
            SphericalImage: visualizable, contains location and circle sample points
        """

        # input validation
        assert num_display_kernel > 0, "must visualize at least one kernel"

        # get radius
        r = np.array([self.radius], dtype=np.float32)

        # gather centers
        if isinstance(getattr(self, "latest_output_vector", None), torch.Tensor):
            kernel_center = ripple_sort(
                to_numpy(copy_or_clone(self.latest_output_vector)), circle_group_center
            )
        else:
            kernel_center = circle_group_center.view(-1, 3)

        kernel_color = colorize_locations(kernel_center)
        kernel_center_spherical_image = SphericalImage(
            value=kernel_color, vector=kernel_center
        )

        # gather center
        circle = kernel_center[:num_display_kernel]

        # construct circle points
        circle_points = sample_circles(circle, r, circle_spacing)
        circle_color = kernel_color[:num_display_kernel, np.newaxis, :].repeat(
            circle_points.shape[1], axis=1
        )
        circle_spherical_image = SphericalImage(
            value=circle_color, vector=circle_points
        )

        output_spherical_image = circle_spherical_image
        if visualize_input_location:
            assert isinstance(
                getattr(self, "latest_input_vector", None), torch.Tensor
            ), "Must run forward at least once to visualize kernel with input vector location"
            _input_vector_vis = ripple_sort(
                to_numpy(copy_or_clone(self.latest_input_vector)), circle_group_center
            )
            input_vector_spherical_image = SphericalImage(
                value=colorize_locations(_input_vector_vis, colormap="spring"),
                vector=_input_vector_vis,
            )  # using a different colormap for differentiation
            output_spherical_image += input_vector_spherical_image
        if visualize_output_location:
            output_spherical_image += kernel_center_spherical_image

        return output_spherical_image


register_cache("CirclePool.output_vector_cache", CirclePool.output_vector_cache)
register_cache("CirclePool.collection_matrix_cache", CirclePool.collection_matrix_cache)
