#!/usr/bin/env python3
#
# Created on Sun Feb 02 2025 16:31:12
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
from opt_einsum import contract
from torch_scatter import scatter

from usf.network.layer.spherical.generic_spherical_cnn import GenericSphericalConv
from usf.sampler.location.location_sampler import LocationSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.spherical import colorize_locations, ripple_sort
from usf.utils.spherical_image import BatchSphericalImage, SphericalImage
from usf.utils.torch_numpy import array_hash, array_size_MiB, copy_or_clone, to_numpy
from usf.visualization.spherical_layer import sample_circles


class CircleConv(GenericSphericalConv):
    """Discrete ring-partitioned spherical convolution.

    Partitions the geodesic disk around each output point into concentric rings
    and learns a weight matrix per ring. Supports normalization modes and
    caches collection matrices at the class level via shared LFU caches.
    """

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
        backend: str = "circle",
    ):
        """ring/radial convolution on spherical signals (S2)

        Args:
            in_channels (int): number of input channels of SphericalImage.value
            out_channels (int): number of output channels of SphericalImage.value
            radius (float): in radians, geodesic radius of the kernel at an output location, analogous to kernel size
            weighting_function_config (dict[str, dict[str, Any]]): Weighting function config passed to discrete embedding.
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
            backend (str, optional): to swallow backend kwarg from parent class initialization. Defaults to "circle".
        """
        assert (
            backend == "circle"
        ), f"wrong backend instantiated, got backend = {backend}"
        super().__init__()

        assert in_channels > 0, "in_channels must be > 0"
        assert out_channels > 0, "out_channels must be > 0"
        assert 0.0 < radius <= torch.pi, f"radius {radius} must be in (0, π]"
        assert resolution_factor > 0.0, "resolution_factor must be > 0.0"
        assert (
            weighting_function_config["distance"]["function"] == "discrete"
        ), "distance function must be discrete when using CircleConv"
        assert (
            weighting_function_config["distance"]["num_slices"] > 0
        ), "num_slices must be > 0"
        assert block_size > 0, "block_size must be positive"

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radius = radius
        self.num_rings = weighting_function_config["distance"]["num_slices"]

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

        # weight shape: (num_rings, in_channels, out_channels)
        self.weight = nn.Parameter(
            torch.empty(self.num_rings, in_channels, out_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()
        self.to(dtype=dtype, device=device)

    def reset_parameters(self) -> CircleConv:
        """Reinitialize weights (Kaiming uniform) and biases (uniform).

        Returns:
            CircleConv: ``self`` for chaining.
        """
        weight_flat = (
            self.weight.permute(2, 0, 1)
            .contiguous()
            .view(self.out_channels, self.num_rings * self.in_channels)
        )  # (out, rings x in)
        nn.init.kaiming_uniform_(weight_flat, a=math.sqrt(5))  # standard fan_in init
        self.weight.data.copy_(
            weight_flat.view(self.out_channels, self.num_rings, self.in_channels)
            .permute(1, 2, 0)
            .contiguous()
        )
        if isinstance(getattr(self, "bias", None), nn.Parameter):
            fan_in = self.num_rings * self.in_channels
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        return self

    def extra_repr(self) -> str:
        """For printing Module information

        Returns:
            str: args and kwargs of the Module
        """
        extra_repr = (
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", radius={self.radius:.5f}"
            f", num_rings={self.num_rings}"
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

    # @torch.compile
    @torch.no_grad()
    def _block_collection_matrix(
        self, in_vector: torch.Tensor, out_vector: torch.Tensor, block_size: int = 2048
    ):
        """chunked collection matrix creation

        Args:
            in_vector (torch.Tensor): shape (I, 3)
            out_vector (torch.Tensor): shape (O, 3)
            block_size (int, optional): size of each chunk. Defaults to 2048.

        Returns:
            torch.Tensor: collection_matrix of shape (3, num_edges), first row indicates input index, second output, third ring membership
        """
        assert out_vector.shape[-1] == in_vector.shape[-1], "last dimension must match"

        # define variables
        _device, _dtype = self.device, self.dtype
        num_out_pixels, num_in_pixels = out_vector.shape[0], in_vector.shape[0]

        # e.g.: tensor([0.0327, 0.0654, 0.0982])
        ring_angles = torch.linspace(
            0,
            self.radius,
            self.num_rings + 1,
            device=_device,
            dtype=_dtype,
        )[1:]
        # e.g.: tensor([0.9995, 0.9979, 0.9952])
        ring_cos = torch.cos(ring_angles)

        # unify device and dtype
        out_vector = out_vector.to(device=_device, dtype=_dtype)
        in_vector = in_vector.to(device=_device, dtype=_dtype)

        # chop up out_vector
        out_vector_chunks = torch.split(out_vector, block_size, dim=0)

        # chop up in_vector
        in_vector_chunks = torch.split(in_vector, block_size, dim=0)

        # record processed columns
        num_processed_columns = 0
        collection_matrix_blocks = []

        for out_vector_chunk in out_vector_chunks:
            num_processed_rows = 0

            for in_vector_chunk in in_vector_chunks:
                # shape (in_vector_chunk.shape[0], out_vector_chunk.shape[0])
                dot_block = in_vector_chunk @ out_vector_chunk.T

                # shape (out_vector_chunk.shape[0], in_vector_chunk.shape[0])
                # element indicates which ring it belongs to, range: [0, num_rings]
                # last means not in any ring centered at this output_vector
                ring_index_block = torch.bucketize(-dot_block, -ring_cos)

                # shape: (K, 2), each row represents [in_idx, out_idx] pairs
                circle_membership_index_block = torch.argwhere(
                    ring_index_block < self.num_rings
                )

                # shape: (K, 1)
                ring_membership_block = ring_index_block[
                    circle_membership_index_block[:, 0],
                    circle_membership_index_block[:, 1],
                ].unsqueeze(-1)

                # Shift row indices for in_vector chunk
                circle_membership_index_block[:, 0] += num_processed_rows
                # Shift column indices for out_vector chunk
                circle_membership_index_block[:, 1] += num_processed_columns

                collection_matrix_block = torch.hstack(
                    (circle_membership_index_block, ring_membership_block)
                )
                collection_matrix_blocks.append(collection_matrix_block)

                num_processed_rows += in_vector_chunk.shape[0]

                del dot_block

            num_processed_columns += out_vector_chunk.shape[0]

        # Concatenate all blocks and transpose, shape (3, E)
        collection_matrix = torch.vstack(collection_matrix_blocks).T

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
            + str(self.num_rings)
            + "_"
            + str(self.radius).replace(".", "dot")  # buffer name doesn't allow '.'
        )
        buffer_name = f"collection_matrix_{input_output_vector_hash}"

        # Get buffer (if it exists)
        buffer_entry = getattr(self, buffer_name, None)

        # If one is missing, recover it from the other
        if (
            input_output_vector_hash not in self.collection_matrix_cache
            and buffer_entry is not None
        ):
            collection_matrix = buffer_entry
            self.collection_matrix_cache[input_output_vector_hash] = collection_matrix
        elif (
            input_output_vector_hash in self.collection_matrix_cache
            and buffer_entry is None
        ):
            collection_matrix = self.collection_matrix_cache[input_output_vector_hash]
            if is_cache_enabled():
                self.register_buffer(buffer_name, collection_matrix)
        elif (
            input_output_vector_hash not in self.collection_matrix_cache
            and buffer_entry is None
        ):
            # Both are missing, so compute and cache the result
            collection_matrix = self._block_collection_matrix(
                input_vector, output_vector, self.block_size
            )
            self.collection_matrix_cache[input_output_vector_hash] = collection_matrix
            if is_cache_enabled():
                self.register_buffer(buffer_name, collection_matrix)
        else:
            collection_matrix = buffer_entry

        return collection_matrix.to(device=self.device)

    # can use @torch.compile on hardware with larger GPU memory
    def forward(
        self, batch_spherical_image: BatchSphericalImage
    ) -> BatchSphericalImage:
        """Main forward pass, has complicated caching mechanism

        Args:
            batch_spherical_image (BatchSphericalImage): batch_value in format (batch_size, num_in_pixels, in_channels)

        Touches:
            self.collection_matrix_cache: for speedup when encountering the same input vector again
            self.output_vector_cache: for speedup when encountering the same input vector again
            self.latest_input_vector: for kernel visualization
            self.latest_output_vector: for kernel visualization
            self.latest_collection_matrix: for metrics reporting

        Returns:
            BatchSphericalImage: batch_value in format (batch_size, num_out_pixels, out_channels)
        """
        input_vector_dtype = batch_spherical_image.vector.dtype

        batch_size, _, num_input_channels = batch_spherical_image.batch_value.shape
        assert (
            num_input_channels == self.in_channels
        ), f"in_channels mismatch: input_spherical_image has {num_input_channels}, but expected {self.in_channels}"

        input_vector = batch_spherical_image.vector.view(-1, 3)
        self.latest_input_vector = input_vector

        output_vector = self._get_output_vector(input_vector)
        self.latest_output_vector = output_vector

        collection_matrix = self._get_collection_matrix(input_vector, output_vector)
        self.latest_collection_matrix = collection_matrix

        input_batch_value = batch_spherical_image.batch_value  # (B, I, C_in)
        num_output_vectors = output_vector.shape[0]
        edge_batch_value = input_batch_value[:, collection_matrix[0]]  # (B, E, C_in)
        collected_batch_value = torch.zeros(
            (batch_size, num_output_vectors * self.num_rings, self.in_channels),
            device=edge_batch_value.device,
            dtype=edge_batch_value.dtype,
        )  # (B, O x ring, C_in)
        scatter(
            src=edge_batch_value,
            index=collection_matrix[1] * self.num_rings + collection_matrix[2],
            dim=-2,
            out=collected_batch_value,
            reduce="sum",
        )
        collected_batch_value = collected_batch_value.reshape(
            -1,
            num_output_vectors,
            self.num_rings,
            self.in_channels,
        )  # (B, O, ring, C_in)
        output_batch_value = contract(
            "bnri,rio->bno",
            collected_batch_value,
            self.weight,
            memory_limit="max_input",
        )  # (B, O, C_out)

        # normalize w.r.t. # input pixels falling into each output pixel
        with torch.no_grad():
            counts = torch.bincount(
                collection_matrix[1], minlength=num_output_vectors
            )  # (O,)
            factor = 1 / counts.clamp_min(1)  # (O,)

        output_batch_value *= factor[None, :, None]  # broadcast to (B, O, C_out)

        if isinstance(getattr(self, "bias", None), nn.Parameter):
            # implicit broadcasting of self.bias happening here
            # (batch_size, num_out_pixels, out_channel)
            output_batch_value += self.bias.view(1, 1, -1)

        return BatchSphericalImage(
            batch_value=output_batch_value,
            vector=output_vector.to(dtype=input_vector_dtype),
        )

    @torch.inference_mode()
    def report_metrics(self, return_metrics: bool = False) -> dict | None:
        """Report statistics on the latest collection matrix for CircleCNN.

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

        collection_matrix = self.latest_collection_matrix
        _device = collection_matrix.device

        in_idx = collection_matrix[0].clone().detach()
        out_idx = collection_matrix[1].clone().detach()
        ring_id = collection_matrix[2].clone().detach()

        num_in_pixels = self.latest_input_vector.shape[0]
        num_out_pixels = self.latest_output_vector.shape[0]
        num_rings = self.num_rings

        # 1) Inputs per (ring, out)
        ring_out_counts = torch.zeros(
            (num_rings, num_out_pixels), dtype=torch.int64, device=_device
        )
        ring_out_counts.index_put_(
            (ring_id, out_idx),
            torch.ones_like(ring_id, dtype=torch.int64),
            accumulate=True,
        )

        avg_input_per_ring = ring_out_counts.float().mean(dim=1)
        std_input_per_ring = ring_out_counts.float().std(dim=1, unbiased=False)
        max_input_per_ring = ring_out_counts.max(dim=1)[0]
        min_input_per_ring = ring_out_counts.min(dim=1)[0]

        # 2) Overall inputs per out
        input_per_out = ring_out_counts.sum(dim=0)
        avg_input_per_output = input_per_out.float().mean().item()
        std_input_per_output = input_per_out.float().std(unbiased=False).item()
        max_input_per_output = input_per_out.max().item()
        min_input_per_output = input_per_out.min().item()

        # 3) Access per input
        input_access_counts = torch.bincount(in_idx, minlength=num_in_pixels)
        accessed = input_access_counts[input_access_counts > 0]
        avg_access_per_input = accessed.float().mean().item()
        std_access_per_input = accessed.float().std(unbiased=False).item()
        max_access_per_input = accessed.max().item()
        min_access_per_input = accessed.min().item()

        # 4) Coverages
        input_coverage = in_idx.unique().numel() / num_in_pixels
        output_coverage = out_idx.unique().numel() / num_out_pixels

        # Report results (Rounding Only in Print)
        print("Circle CNN Metrics:")
        print(
            f"\tAvg/Std/Max/Min # input pixels per ring per output pixel: "
            f"{[f'{round(a.item(), 3)}/{round(s.item(), 3)}/{int(mx)}/{int(mn)}' for a, s, mx, mn in zip(avg_input_per_ring, std_input_per_ring, max_input_per_ring, min_input_per_ring)]}"
        )
        print(
            f"\tAvg/Std/Max/Min # input pixels per output pixel: "
            f"{round(avg_input_per_output, 3)}/{round(std_input_per_output, 3)}/{max_input_per_output}/{min_input_per_output}"
        )
        print(
            f"\tAvg/Std/Max/Min # access per input pixel: "
            f"{round(avg_access_per_input, 3)}/{round(std_access_per_input, 3)}/{max_access_per_input}/{min_access_per_input}"
        )
        print(
            f"\tInput/Output pixel coverage: {round(input_coverage * 100, 1)}%/{round(output_coverage * 100, 1)}%"
        )
        print(
            f"\tCollection matrix shape/size: {collection_matrix.shape}/{round(array_size_MiB(collection_matrix), 3)} MiB"
        )

        return (
            {
                "avg_input_per_ring": avg_input_per_ring,
                "std_input_per_ring": std_input_per_ring,
                "max_input_per_ring": max_input_per_ring,
                "min_input_per_ring": min_input_per_ring,
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
        ring_group_center: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32),
        ring_spacing: float = 2 * np.pi / 5000,
        only_outer_ring: bool = False,
    ) -> SphericalImage:
        """visualize the output location and several rings

        Args:
            visualize_input_location (bool, optional): whether to see the latest input location. Defaults to True.
            visualize_output_location (bool, optional): whether to see the latest output location. Defaults to True.
            num_display_kernel (int, optional): number of ring center locations, e.g. the first 5. Defaults to 5.
            ring_group_center (np.ndarray, optional): in cartesian coordinate, the point where center location are around. Defaults to np.array([-1, 0, 0])
            ring_spacing (float, optional): spacing between 2 sample points on a ring. Defaults to 2 * np.pi / 5000.
            only_outer_ring (bool, optional): only visualize the outer-most ring for each center. Defaults to False.

        Returns:
            SphericalImage: visualizable, contains location and ring sample points
        """

        # input validation
        assert num_display_kernel > 0, "must visualize at least one kernel"

        # gather geodesic radii
        num_vis_rings = 1 if only_outer_ring else self.num_rings
        r = np.linspace(0, self.radius, num_vis_rings + 1, dtype=np.float32)[1:]

        # gather centers
        if isinstance(getattr(self, "latest_output_vector", None), torch.Tensor):
            kernel_center = ripple_sort(
                to_numpy(copy_or_clone(self.latest_output_vector)), ring_group_center
            )
        else:
            kernel_center = ring_group_center.view(-1, 3)

        kernel_color = colorize_locations(kernel_center)
        kernel_center_spherical_image = SphericalImage(
            value=kernel_color, vector=kernel_center
        )

        # gather center
        ring_center = kernel_center[:num_display_kernel]

        # construct ring points
        ring_points = sample_circles(ring_center, r, ring_spacing)
        ring_color = kernel_color[:num_display_kernel, np.newaxis, :].repeat(
            ring_points.shape[1], axis=1
        )
        ring_spherical_image = SphericalImage(value=ring_color, vector=ring_points)

        output_spherical_image = ring_spherical_image
        if visualize_input_location:
            assert isinstance(
                getattr(self, "latest_input_vector", None), torch.Tensor
            ), "Must run forward at least once to visualize kernel with input vector location"
            _input_vector_vis = ripple_sort(
                to_numpy(copy_or_clone(self.latest_input_vector)), ring_group_center
            )
            input_vector_spherical_image = SphericalImage(
                value=colorize_locations(_input_vector_vis, colormap="spring"),
                vector=_input_vector_vis,
            )  # using a different colormap for differentiation
            output_spherical_image += input_vector_spherical_image
        if visualize_output_location:
            output_spherical_image += kernel_center_spherical_image

        return output_spherical_image


register_cache("CircleConv.output_vector_cache", CircleConv.output_vector_cache)
register_cache("CircleConv.collection_matrix_cache", CircleConv.collection_matrix_cache)
