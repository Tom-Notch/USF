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
from torch_scatter import scatter

from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.spherical import colorize_locations, ripple_sort
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import array_hash, array_size_MiB, copy_or_clone, to_numpy
from usf.visualization.spherical_layer import sample_circles


class CirclePool(nn.Module):
    """Geodesic circle-based pooling on spherical signals (S²).

    For each output location, collects all input pixels within a geodesic
    radius, then reduces them via the selected ``pool_type`` (max, min, or
    mean). Collection matrices are cached at the class level in shared LFU
    caches to avoid redundant re-computation across instances and forward
    calls.
    """

    # class level cache shared among instances, need to guarantee uniqueness
    output_vector_cache = LFUCache(maxsize=500)
    collection_matrix_cache = LFUCache(maxsize=500)

    def __init__(
        self,
        *,
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
            radius (float): in radians, geodesic radius of the pooling circle at an output location, analogous to kernel size
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

        assert 0.0 < radius <= torch.pi, f"radius {radius} must be in (0, π]"
        assert (
            resolution_factor > 0.0
        ), f"resolution_factor {resolution_factor} must be > 0.0"
        assert block_size > 0, "block_size must be positive"

        self.pool_type = pool_type

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
        self, in_vector: torch.Tensor, out_vector: torch.Tensor, block_size: int = 2048
    ) -> torch.Tensor:
        """Blocked collection matrix creation for circle pooling.

        Args:
            in_vector (torch.Tensor): shape (I, 3)
            out_vector (torch.Tensor): shape (O, 3)
            block_size (int, optional): Size of each block for chunking. Defaults to 2048.

        Returns:
            collection_matrix (torch.Tensor): shape (2, num_edges), first row indicates input index, second output
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

        # Chunk out_vector
        out_vector_chunks = torch.split(out_vector, block_size, dim=0)

        # Chunk in_vector
        in_vector_chunks = torch.split(in_vector, block_size, dim=0)

        # Track processed rows/cols
        num_processed_columns = 0
        circle_membership_index_blocks = []

        for out_vector_chunk in out_vector_chunks:
            num_processed_rows = 0

            for in_vector_chunk in in_vector_chunks:
                # Compute dot product
                # shape: (in_vector_chunk.shape[0], out_vector_chunk.shape[0])
                dot_block = in_vector_chunk @ out_vector_chunk.T

                # Determine membership (binary mask)
                # shape: (K, 2), each row represents [in_idx, out_idx] pairs
                circle_membership_index_block = torch.argwhere(dot_block >= circle_cos)

                # Shift row indices for in_vector chunk
                circle_membership_index_block[:, 0] += num_processed_rows
                # Shift column indices for out_vector chunk
                circle_membership_index_block[:, 1] += num_processed_columns

                # Store block
                circle_membership_index_blocks.append(circle_membership_index_block)
                num_processed_rows += in_vector_chunk.shape[0]

                del dot_block

            num_processed_columns += out_vector_chunk.shape[0]

        # Concatenate all blocks and transpose, shape (2, E)
        collection_matrix = torch.vstack(circle_membership_index_blocks).T

        return collection_matrix

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
    ) -> torch.Tensor:
        # class level uniqueness guaranteed
        input_output_vector_hash = (
            array_hash(torch.vstack((input_vector, output_vector)))
            + "_"
            + str(self.radius).replace(".", "dot")  # buffer name doesn't allow '.'
        )

        # Get cache and buffer (if they exist)
        collection_matrix_buffer_entry = getattr(self, input_output_vector_hash, None)

        # If one is missing, recover it from the other
        if (
            input_output_vector_hash not in self.collection_matrix_cache
            and collection_matrix_buffer_entry is not None
        ):
            collection_matrix = collection_matrix_buffer_entry
            self.collection_matrix_cache[input_output_vector_hash] = collection_matrix
        elif (
            input_output_vector_hash in self.collection_matrix_cache
            and collection_matrix_buffer_entry is None
        ):
            collection_matrix = self.collection_matrix_cache[input_output_vector_hash]
            if is_cache_enabled():
                self.register_buffer(input_output_vector_hash, collection_matrix)
        elif (
            input_output_vector_hash not in self.collection_matrix_cache
            and collection_matrix_buffer_entry is None
        ):
            # Both are missing, so compute and cache the result
            collection_matrix = self._block_collection_matrix(
                input_vector, output_vector, self.block_size
            )

            self.collection_matrix_cache[input_output_vector_hash] = collection_matrix
            if is_cache_enabled():
                self.register_buffer(input_output_vector_hash, collection_matrix)
        else:
            collection_matrix = collection_matrix_buffer_entry

        return collection_matrix.to(device=self.device)

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

        collection_matrix = self._get_collection_matrix(input_vector, output_vector)
        self.latest_collection_matrix = collection_matrix  # for metrics report

        input_batch_value = batch_spherical_image.batch_value  # (B, I, C)
        B, I, C = input_batch_value.shape
        O = output_vector.shape[0]
        edge_batch_value = input_batch_value[:, collection_matrix[0]]  # (B, E, C)
        output_batch_value = torch.zeros(
            (B, O, C), device=_device, dtype=_dtype
        )  # explicit initialization to prevent inf/-inf in unfilled slots
        scatter(
            src=edge_batch_value,
            index=collection_matrix[1],
            dim=-2,
            out=output_batch_value,
            reduce=self.pool_type,
        )  # (B, O, C)

        return BatchSphericalImage(batch_value=output_batch_value, vector=output_vector)

    @torch.inference_mode()
    def report_metrics(self, return_metrics: bool = False) -> dict | None:
        """Report statistics on the latest pooling collection matrix."""

        if getattr(self, "latest_collection_matrix", None) is None:
            warnings.warn(
                "No collection matrix found. Run forward() first.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # Shapes
        num_in_pixels = self.latest_input_vector.shape[0]
        num_out_pixels = self.latest_output_vector.shape[0]

        collection_matrix = self.latest_collection_matrix  # (2, E)

        in_idx = collection_matrix[0]
        out_idx = collection_matrix[1]

        # --- 1) #input pixels contributing to each output pixel
        input_per_output = torch.bincount(out_idx, minlength=num_out_pixels).float()
        avg_input_per_output = input_per_output.mean().item()
        std_input_per_output = input_per_output.std(unbiased=False).item()
        max_input_per_output = int(input_per_output.max().item())
        min_input_per_output = int(input_per_output.min().item())

        # --- 2) Fraction of input pixels that are used at least once
        unique_inputs = torch.unique(in_idx).numel()
        input_coverage = (
            float(unique_inputs) / float(num_in_pixels) if num_in_pixels > 0 else 0.0
        )

        # --- 3) Fraction of outputs that receive at least one input
        output_coverage = (
            (input_per_output > 0).float().mean().item() if num_out_pixels > 0 else 0.0
        )

        # --- 4) Access counts per input pixel (how many outputs each input contributes to)
        input_access_counts = torch.bincount(in_idx, minlength=num_in_pixels).float()
        accessed_inputs = input_access_counts[input_access_counts > 0]

        avg_access_per_input = accessed_inputs.mean().item()
        std_access_per_input = accessed_inputs.std(unbiased=False).item()
        max_access_per_input = int(accessed_inputs.max().item())
        min_access_per_input = int(accessed_inputs.min().item())

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
            f"\tcollection_matrix shape/size: {collection_matrix.shape}/{round(array_size_MiB(collection_matrix), 3)} MiB"
        )

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
