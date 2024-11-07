#!/usr/bin/env python3
#
# Created on Wed Oct 1 2025 16:40:13
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import warnings
from typing import Any

import numpy as np
import torch
from cachetools import LFUCache
from multimethod import multimethod
from torch_scatter import scatter

from usf.sampler.value.value_sampler import ValueSampler
from usf.utils.cache import is_cache_enabled, register_cache
from usf.utils.nearest_neighbor import nearest_point
from usf.utils.spherical_image import BatchSphericalImage
from usf.utils.torch_numpy import array_hash, array_size_MiB, to_numpy, to_torch


# use cuda whenever possible in _block_collection_matrix
def _pick_exec_device(
    tensors: list[torch.Tensor],
    prefer_device: torch.device | None = None,
) -> torch.device:
    """
    Device policy:
      1) If a tensor is CUDA -> pick that CUDA device.
      2) Else if prefer_device is CUDA -> use it.
      3) Else if CUDA is available -> use current CUDA device.
      4) Else CPU.
    """
    for tensor in tensors:
        if tensor.is_cuda:
            return tensor.device

    if prefer_device is not None:
        return prefer_device

    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")

    return torch.device("cpu")


class RadialBasisFunction(ValueSampler):
    """Radial Basis Function: output is only dependent on distance to input points inside a circle"""

    index_distance_cache = LFUCache(maxsize=500)
    radius_cache = LFUCache(maxsize=500)

    def __init__(self, config: dict, *args, **kwargs) -> None:
        """Initialize RBF value sampler.

        Args:
            config (dict): Must contain ``"value_sampler_config"`` with optional
                keys ``num_in_circle_points`` (int, default 4),
                ``radius`` (float | None), ``block_size`` (int, default 8192),
                ``kernel`` (str, default ``"gaussian"``),
                ``sigma`` (float, default 0.2), ``eps`` (float, default 1e-10).
        """
        super().__init__(config, *args, **kwargs)
        self.value_sampler_config: dict[str, Any] = config.get(
            "value_sampler_config", {}
        )
        self.num_in_circle_points: int = self.value_sampler_config.get(
            "num_in_circle_points", 4
        )
        self.radius: float | None = self.value_sampler_config.get("radius", None)
        self.block_size: int = self.value_sampler_config.get("block_size", 8192)
        self.kernel: str = self.value_sampler_config.get("kernel", "gaussian")
        self.sigma: float = self.value_sampler_config.get("sigma", 0.2)
        self.eps: float = self.value_sampler_config.get("eps", 1e-10)

    def extra_args(self) -> str:
        """overridden function to provide more args to be appended to extra_repr

        Returns:
            str: extra args string
        """
        extra_args = ""

        if self.radius is None:
            extra_args += f"num_in_circle_points={self.num_in_circle_points}"
        else:
            extra_args += f"radius={self.radius:.2f}"

        extra_args += f", kernel={self.kernel}"

        if self.kernel == "gaussian":
            extra_args += f", sigma={self.sigma:.2f}"

        extra_args += f", block_size={self.block_size}"

        return extra_args

    @staticmethod
    @torch.no_grad()
    def _block_collection_matrix(
        input_vectors: torch.Tensor,
        output_vectors: torch.Tensor,
        radius: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (
            output_vectors.shape[-1] == input_vectors.shape[-1]
        ), "Last dimension must match"

        original_device = input_vectors.device
        _device = _pick_exec_device([input_vectors, output_vectors])
        _dtype = input_vectors.dtype

        # Unify device and dtype
        output_vectors = output_vectors.to(device=_device, dtype=_dtype)
        input_vectors = input_vectors.to(device=_device, dtype=_dtype)

        output_vectors.shape[0]
        input_vectors.shape[0]

        cosine_threshold = torch.cos(torch.tensor(radius, device=_device, dtype=_dtype))

        membership_indices_list: list[torch.Tensor] = []
        relative_distances_list: list[torch.Tensor] = []

        # Chunk
        output_chunks = torch.split(output_vectors, block_size, dim=0)
        input_chunks = torch.split(input_vectors, block_size, dim=0)

        output_offset = 0
        for output_chunk in output_chunks:
            input_offset = 0
            for input_chunk in input_chunks:
                # shape: (chunk_input_pixels, chunk_output_pixels)
                dot_block = torch.clamp(input_chunk @ output_chunk.T, -1.0, 1.0)
                valid_mask_block = dot_block >= cosine_threshold
                if valid_mask_block.sum() > 0:
                    valid_indices = torch.nonzero(
                        valid_mask_block
                    )  # shape: (num_valid, 2)
                    valid_indices[:, 0] += input_offset
                    valid_indices[:, 1] += output_offset
                    membership_indices_list.append(valid_indices.T)  # (2, num_valid)

                    relative_distances = torch.acos(
                        dot_block[valid_mask_block]
                    ).unsqueeze(
                        0
                    )  # (1, num_valid)
                    relative_distances_list.append(relative_distances)

                del dot_block

                input_offset += input_chunk.shape[0]

            output_offset += output_chunk.shape[0]

        indices = torch.hstack(membership_indices_list)
        distances = torch.hstack(relative_distances_list)

        # Normalize distance to [0.0, 1.0]
        distances /= radius
        distances.clamp_(0.0, 1.0)

        return indices.to(device=original_device), distances.to(device=original_device)

    @multimethod
    @torch.no_grad()
    def sample_value(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: np.ndarray,
        *,
        num_in_circle_points: int | None = None,
        radius: float | None = None,
        block_size: int | None = None,
        kernel: str | None = None,
        sigma: float | None = None,
        eps: float | None = None,
    ) -> np.ndarray:
        """sample value using RBF kernels

        Args:
            batch_spherical_image (BatchSphericalImage): input batch spherical image
            sample_vector (np.ndarray): target output vector
            num_in_circle_points (int | None, optional): number of desired points inside a circle. Defaults to None.
            radius (float | None, optional): radius override, will invalidate num_in_circle_points if provided. Defaults to None.
            block_size (int | None, optional): block size for _block_collection_matrix, tradeoff speed and memory. Defaults to None.
            kernel (str | None, optional): RBF kernels, currently support {"gaussian", "hann", "wendland"}. Defaults to None.
            sigma (float | None, optional): for gaussian kernel, analogous to temperature, smaller results in sharper interpolation. Defaults to None.
            eps (float | None, optional): small number for numerical stability. Defaults to None.

        Raises:
            NotImplementedError: when kernel is not implemented

        Returns:
            np.ndarray: output value at output vector
        """
        assert isinstance(
            batch_spherical_image.batch_value, np.ndarray
        ), "Expect input BatchSphericalImage to use np.ndarray backend, same to sample_vector"

        eps = self.eps if eps is None else eps

        input_batch_value = batch_spherical_image.batch_value
        B, _, C = input_batch_value.shape
        O = sample_vector.shape[0]

        image_vector = batch_spherical_image.vector  # shape (I, 3)

        # for report_metrics
        self.latest_input_vector = to_torch(image_vector).clone()
        self.latest_output_vector = to_torch(sample_vector).clone()

        if radius is None:
            if self.radius is not None:
                radius = self.radius
            else:
                # determine plausible radius given desired number of in circle points
                num_in_circle_points = (
                    self.num_in_circle_points
                    if num_in_circle_points is None
                    else num_in_circle_points
                )
                radius_cache_key = f"{array_hash(image_vector)}_{num_in_circle_points}"
                if radius_cache_key in self.radius_cache:
                    radius = self.radius_cache[radius_cache_key]
                else:
                    _, angles = nearest_point(
                        image_vector,
                        image_vector,
                        num_in_circle_points + 1,
                    )
                    radius = angles[:, -1].mean().item()
                    self.radius_cache[radius_cache_key] = radius

        self.latest_radius = radius

        block_size = (
            self.block_size if block_size is None else block_size
        )  # default is 8192

        input_output_vector_hash = (
            f"{array_hash(image_vector)}_{array_hash(sample_vector)}"
        )
        geometry_cache_key = f"RBF_{input_output_vector_hash}_{radius}".replace(
            ".", "dot"
        )
        index_buffer_name = f"RBF_index_{input_output_vector_hash}_{radius}".replace(
            ".", "dot"
        )
        distance_buffer_name = (
            f"RBF_distance_{input_output_vector_hash}_{radius}".replace(".", "dot")
        )

        index_buffer_entry = getattr(self, index_buffer_name, None)
        distance_buffer_entry = getattr(self, distance_buffer_name, None)

        if (
            geometry_cache_key not in self.index_distance_cache
            and index_buffer_entry is not None
        ):
            index, distance = index_buffer_entry, distance_buffer_entry
            self.index_distance_cache[geometry_cache_key] = index, distance
        elif (
            geometry_cache_key in self.index_distance_cache
            and index_buffer_entry is None
        ):
            index, distance = self.index_distance_cache[geometry_cache_key]
            if is_cache_enabled():
                self.register_buffer(index_buffer_name, index)
                self.register_buffer(distance_buffer_name, distance)
        elif (
            geometry_cache_key not in self.index_distance_cache
            and index_buffer_entry is None
        ):
            index, distance = self._block_collection_matrix(
                to_torch(image_vector),
                to_torch(sample_vector),
                radius=radius,
                block_size=block_size,
            )

            self.index_distance_cache[geometry_cache_key] = index, distance

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, index)
                self.register_buffer(distance_buffer_name, distance)
        else:
            index, distance = index_buffer_entry, distance_buffer_entry

        index, distance = to_numpy(index), to_numpy(distance)

        # for report_metrics
        self.latest_index = to_torch(index).clone()
        self.latest_distance = to_torch(distance).clone()

        kernel = self.kernel if kernel is None else kernel  # default is gaussian
        if kernel == "gaussian":
            sigma = self.sigma if sigma is None else sigma
            weight = np.exp(-0.5 * (distance / sigma) ** 2)
        elif kernel == "hann":
            weight = 0.5 * (1.0 + np.cos(np.pi * distance))
        elif kernel == "wendland":
            weight = ((1 - distance) ** 4) * (1 + 4 * distance)
        else:
            raise NotImplementedError(f"RBF kernel {kernel} not implemented")
        weight = weight.squeeze()

        # normalize weight so that sum of weight at an output vector is 1
        weight_sum = np.zeros((O), dtype=weight.dtype)  # shape (O,)
        np.add.at(weight_sum, index[1], weight)
        normalized_weight = weight / (weight_sum[index[1]] + eps)

        edge_batch_value = input_batch_value[:, index[0]].astype(
            weight.dtype
        )  # shape (B, E, C)
        weighted_edge_batch_value = (
            edge_batch_value * normalized_weight[None, :, None]
        )  # shape (B, E, C)

        # sum up
        sampled_colors = np.zeros((B, O, C), dtype=weighted_edge_batch_value.dtype)
        for b in range(B):  # to mimic scatter with reduce="sum"
            np.add.at(
                sampled_colors[b], (index[1], slice(None)), weighted_edge_batch_value[b]
            )

        return sampled_colors.astype(input_batch_value.dtype)

    @multimethod
    def sample_value(
        self,
        batch_spherical_image: BatchSphericalImage,
        sample_vector: torch.Tensor,
        *,
        num_in_circle_points: int | None = None,
        radius: float | None = None,
        block_size: int | None = None,
        kernel: str | None = None,
        sigma: float | None = None,
        eps: float | None = None,
    ) -> torch.Tensor:
        """sample value using RBF kernels

        Args:
            batch_spherical_image (BatchSphericalImage): input batch spherical image
            sample_vector (torch.Tensor): target output vector
            num_in_circle_points (int | None, optional): number of desired points inside a circle. Defaults to None.
            radius (float | None, optional): radius override, will invalidate num_in_circle_points if provided. Defaults to None.
            block_size (int | None, optional): block size for _block_collection_matrix, tradeoff speed and memory. Defaults to None.
            kernel (str | None, optional): RBF kernels, currently support {"gaussian", "hann", "wendland"}. Defaults to None.
            sigma (float | None, optional): for gaussian kernel, analogous to temperature, smaller results in sharper interpolation. Defaults to None.
            eps (float | None, optional): small number for numerical stability. Defaults to None.

        Raises:
            NotImplementedError: when kernel is not implemented

        Returns:
            torch.Tensor: output value at output vector
        """
        assert isinstance(
            batch_spherical_image.batch_value, torch.Tensor
        ), "Expect input BatchSphericalImage to use torch.Tensor backend, same to sample_vector"

        eps = self.eps if eps is None else eps

        input_batch_value = batch_spherical_image.batch_value
        _device = input_batch_value.device
        B, _, C = input_batch_value.shape
        O = sample_vector.shape[0]

        image_vector = batch_spherical_image.vector  # shape (I, 3)

        # for report_metrics
        self.latest_input_vector = image_vector.clone()
        self.latest_output_vector = sample_vector.clone()

        if radius is None:
            if self.radius is not None:
                radius = self.radius
            else:
                # determine plausible radius given desired number of in circle points
                num_in_circle_points = (
                    self.num_in_circle_points
                    if num_in_circle_points is None
                    else num_in_circle_points
                )
                radius_cache_key = f"{array_hash(image_vector)}_{num_in_circle_points}"
                if radius_cache_key in self.radius_cache:
                    radius = self.radius_cache[radius_cache_key]
                else:
                    _, angles = nearest_point(
                        image_vector,
                        image_vector,
                        num_in_circle_points + 1,
                    )
                    radius = angles[:, -1].mean().item()
                    self.radius_cache[radius_cache_key] = radius

        self.latest_radius = radius

        block_size = (
            self.block_size if block_size is None else block_size
        )  # default is 8192

        input_output_vector_hash = (
            f"{array_hash(image_vector)}_{array_hash(sample_vector)}"
        )
        geometry_cache_key = f"RBF_{input_output_vector_hash}_{radius}".replace(
            ".", "dot"
        )
        index_buffer_name = f"RBF_index_{input_output_vector_hash}_{radius}".replace(
            ".", "dot"
        )
        distance_buffer_name = (
            f"RBF_distance_{input_output_vector_hash}_{radius}".replace(".", "dot")
        )

        index_buffer_entry = getattr(self, index_buffer_name, None)
        distance_buffer_entry = getattr(self, distance_buffer_name, None)

        if (
            geometry_cache_key not in self.index_distance_cache
            and index_buffer_entry is not None
        ):
            index, distance = index_buffer_entry, distance_buffer_entry
            self.index_distance_cache[geometry_cache_key] = index, distance
        elif (
            geometry_cache_key in self.index_distance_cache
            and index_buffer_entry is None
        ):
            index, distance = self.index_distance_cache[geometry_cache_key]
            if is_cache_enabled():
                self.register_buffer(index_buffer_name, index)
                self.register_buffer(distance_buffer_name, distance)
        elif (
            geometry_cache_key not in self.index_distance_cache
            and index_buffer_entry is None
        ):
            index, distance = self._block_collection_matrix(
                image_vector,
                sample_vector,
                radius=radius,
                block_size=block_size,
            )

            self.index_distance_cache[geometry_cache_key] = index, distance

            if is_cache_enabled():
                self.register_buffer(index_buffer_name, index)
                self.register_buffer(distance_buffer_name, distance)
        else:
            index, distance = index_buffer_entry, distance_buffer_entry

        index, distance = index.to(device=_device), distance.to(device=_device)

        # for report_metrics
        self.latest_index = index.clone()
        self.latest_distance = distance.clone()

        with torch.no_grad():
            kernel = self.kernel if kernel is None else kernel  # default is gaussian
            if kernel == "gaussian":
                sigma = self.sigma if sigma is None else sigma
                weight = torch.exp(-0.5 * (distance / sigma) ** 2)
            elif kernel == "hann":
                weight = 0.5 * (1.0 + torch.cos(torch.pi * distance))
            elif kernel == "wendland":
                weight = ((1 - distance) ** 4) * (1 + 4 * distance)
            else:
                raise NotImplementedError(f"RBF kernel {kernel} not implemented")
            weight = weight.squeeze()

            # normalize weight so that sum of weight at an output vector is 1
            weight_sum = torch.zeros(
                (O), dtype=weight.dtype, device=weight.device
            )  # shape (O,)
            scatter(src=weight, index=index[1], dim=0, out=weight_sum, reduce="sum")
            normalized_weight = weight / (weight_sum[index[1]] + eps)

        edge_batch_value = input_batch_value[:, index[0]].to(
            dtype=weight.dtype
        )  # shape (B, E, C)
        weighted_edge_batch_value = (
            edge_batch_value * normalized_weight[None, :, None]
        )  # shape (B, E, C)

        # sum up
        sampled_colors = torch.zeros(
            (B, O, C),
            dtype=weighted_edge_batch_value.dtype,
            device=_device,
        )
        scatter(
            src=weighted_edge_batch_value,
            index=index[1],
            dim=-2,
            out=sampled_colors,
            reduce="sum",
        )

        return sampled_colors.to(dtype=input_batch_value.dtype)

    @torch.inference_mode()
    def report_metrics(self, return_metrics: bool = False) -> dict | None:
        """Report statistics on the latest collection matrix for SphericalConv."""
        if getattr(self, "latest_index", None) is None:
            warnings.warn(
                "No index found. Run forward() first.",
                UserWarning,
                stacklevel=2,
            )
            return None

        indices = self.latest_index
        in_idx = indices[0]
        out_idx = indices[1]
        distances = self.latest_distance  # normalized geodesic distance in [0.0, 1.0]

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

        avg_normalized_distance = distances.mean().item()
        std_normalized_distance = distances.std(unbiased=False).item()
        max_normalized_distance = distances.max().item()
        min_normalized_distance = distances.min().item()

        # Total number of queries to weighting function
        total_edges = indices.shape[1]

        print("Radial Basis Function Value Sampler Metrics:")
        print(f"\tRadius (in radians): {round(self.latest_radius, 3)}")
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
            f"\tInput/Output pixel coverage: {round(input_coverage * 100, 1)}%/{round(output_coverage * 100, 1)}%"
        )
        print(
            f"\tindex shape/size: {indices.shape}/{round(array_size_MiB(indices), 3)} MiB"
        )
        print(
            f"\tdistance shape/size: {distances.shape}/{round(array_size_MiB(distances), 3)} MiB"
        )
        print(f"\tTotal number of edges: {total_edges}")

        if return_metrics:
            return {
                "radius": self.latest_radius,
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
                "input_coverage": input_coverage,
                "output_coverage": output_coverage,
                "total_edges": total_edges,
            }
        else:
            return None


register_cache(
    "ValueSampler.RadialBasisFunction.index_distance_cache",
    RadialBasisFunction.index_distance_cache,
)
register_cache(
    "ValueSampler.RadialBasisFunction.radius_cache",
    RadialBasisFunction.radius_cache,
)
