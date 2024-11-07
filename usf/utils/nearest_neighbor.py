#!/usr/bin/env python3
#
# Created on Fri Mar 28 2025 15:13:01
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
import numpy as np
import torch
from cachetools import LFUCache
from multimethod import multimethod

from usf.utils.cache import register_cache
from usf.utils.torch_numpy import array_hash, to_numpy, to_torch

supported_backend: list[str] = ["faiss", "hnswlib"]
index_cache: dict[str, LFUCache] = {
    backend: LFUCache(maxsize=500) for backend in supported_backend
}
for backend, cache in index_cache.items():
    register_cache(f"{backend}_index_cache", cache)

result_cache = LFUCache(maxsize=1000)
register_cache("nearest_neighbor_result_cache", result_cache)


def _get_gpu_index() -> int:
    """Get the  GPU index for FAISS, with load balancing across available GPUs.

    Returns:
        int: GPU index for FAISS (-1 for CPU, otherwise logical device index)
    """
    if not torch.cuda.is_available():
        return -1  # Return -1 for CPU case

    # Initialize static counter if not exists
    if not hasattr(_get_gpu_index, "counter"):
        _get_gpu_index.counter = 0

    # Implement round-robin load balancing
    _get_gpu_index.counter = (_get_gpu_index.counter + 1) % torch.cuda.device_count()

    return _get_gpu_index.counter


@multimethod
@torch.no_grad()
def nearest_point(  # type: ignore
    source: torch.Tensor,
    target: torch.Tensor,
    num_neighbors: int,
    backend: str = "faiss",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find nearest num_neighbors target points near source points

    Args:
        source (torch.Tensor): source points for search
        target (torch.Tensor): target points to build index
        num_neighbors (int): number of nearest neighbor
        backend (str, optional): nearest neighbor backend. Defaults to "faiss".

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            neighbor_indices: shape (source.shape[0], num_neighbors), indicates neighbor_indices of target points that are close to source point i
            angles: shape (source.shape[0], num_neighbors), corresponding angular geodesic distance in radian
    """
    assert (
        num_neighbors > 0
    ), f"num_neighbors for nearest_point must be positive, got{num_neighbors}"
    assert (
        backend in supported_backend
    ), f"Supported backends for nearest_point: {supported_backend}, got {backend}"
    assert (
        source.device == target.device
    ), f"source and target must be on the same device, got source on {source.device} and target on {target.device}"

    _device = target.device
    result_hash = array_hash(torch.vstack((source, target))) + "_" + str(num_neighbors)

    # see if result hashed before
    if result_cache.get(result_hash) is None:
        index_hash = array_hash(target)

        if backend == "hnswlib":
            # see if index built before
            if index_cache[backend].get(index_hash) is None:
                import hnswlib

                index = hnswlib.Index(space="cosine", dim=target.shape[-1])
                index.init_index(
                    max_elements=target.shape[0],
                    ef_construction=200,
                    M=64,
                )
                index.add_items(to_numpy(target))
                index.set_ef(200)  # ef for search accuracy

                # cache index
                index_cache[backend][index_hash] = index
            else:
                index = index_cache[backend][index_hash]

            neighbor_indices, cosine_distances = index.knn_query(
                to_numpy(source), k=num_neighbors
            )

            cosines = np.clip(1 - cosine_distances, -1.0, 1.0)
            angles = np.arccos(cosines)

        elif backend == "faiss":
            # see if index built before
            if index_cache[backend].get(index_hash) is None:
                import faiss

                num_clusters = min(
                    128, max(1, target.shape[0] // 40)
                )  # number of clusters

                if torch.cuda.is_available():
                    # use GPU
                    gpu_resource = faiss.StandardGpuResources()
                    # index = faiss.GpuIndexFlatIP(gpu_resource, target.shape[-1])  # exact neighbor indices and distances
                    gpu_config = faiss.GpuIndexIVFFlatConfig()
                    # Get physical GPU index with load balancing
                    gpu_config.device = _get_gpu_index()
                    index = faiss.GpuIndexIVFFlat(
                        gpu_resource,
                        target.shape[-1],
                        num_clusters,
                        faiss.METRIC_INNER_PRODUCT,
                        gpu_config,
                    )  # approximate neighbor indices and exact distances
                else:
                    # use CPU
                    # index = faiss.IndexFlatIP(target.shape[-1])  # exact neighbor indices and distances
                    index = faiss.IndexIVFFlat(
                        faiss.IndexFlatIP(target.shape[-1]),
                        target.shape[-1],
                        num_clusters,
                        faiss.METRIC_INNER_PRODUCT,
                    )  # approximate neighbor indices and exact distances

                np_target = to_numpy(target)
                index.train(np_target)
                index.add(np_target)

                # cache index
                index_cache[backend][index_hash] = index
            else:
                index = index_cache[backend][index_hash]

            dots, neighbor_indices = index.search(to_numpy(source), k=num_neighbors)

            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)

        # cache result
        result_cache[result_hash] = to_torch(
            neighbor_indices, device=_device
        ), to_torch(angles, device=_device)
    else:
        neighbor_indices, angles = result_cache[result_hash]

    return (
        to_torch(neighbor_indices, device=_device),
        to_torch(angles, device=_device),
    )


@multimethod
@torch.no_grad()
def nearest_point(
    source: np.ndarray,
    target: np.ndarray,
    num_neighbors: int,
    backend: str = "faiss",
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest num_neighbors target points near source points

    Args:
        source (np.ndarray): source points for search
        target (np.ndarray): target points to build index
        num_neighbors (int): number of nearest neighbor
        backend (str, optional): nearest neighbor backend. Defaults to "faiss".

    Returns:
        tuple[np.ndarray, np.ndarray]:
            neighbor_indices: shape (source.shape[0], num_neighbors), indicates neighbor_indices of target points that are close to source point i
            angles: shape (source.shape[0], num_neighbors), corresponding angular geodesic distance in radian
    """
    assert (
        num_neighbors > 0
    ), f"num_neighbors for nearest_point must be positive, got{num_neighbors}"
    assert (
        backend in supported_backend
    ), f"Supported backends for nearest_point: {supported_backend}, got {backend}"

    result_hash = array_hash(np.vstack((source, target))) + "_" + str(num_neighbors)

    # see if result hashed before
    if result_cache.get(result_hash) is None:
        index_hash = array_hash(target)

        if backend == "hnswlib":
            # see if index built before
            if index_cache[backend].get(index_hash) is None:
                import hnswlib

                index = hnswlib.Index(space="cosine", dim=target.shape[-1])
                index.init_index(
                    max_elements=target.shape[0],
                    ef_construction=200,
                    M=64,
                )
                index.add_items(target)
                index.set_ef(200)  # ef for search accuracy

                # cache index
                index_cache[backend][index_hash] = index
            else:
                index = index_cache[backend][index_hash]

            neighbor_indices, cosine_distances = index.knn_query(
                source, k=num_neighbors
            )

            cosines = np.clip(1 - cosine_distances, -1.0, 1.0)
            angles = np.arccos(cosines)

        elif backend == "faiss":
            # see if index built before
            if index_cache[backend].get(index_hash) is None:
                import faiss

                num_clusters = min(
                    128, max(1, target.shape[0] // 40)
                )  # number of clusters

                if torch.cuda.is_available():
                    # use GPU
                    gpu_resource = faiss.StandardGpuResources()
                    # index = faiss.GpuIndexFlatIP(gpu_resource, target.shape[-1])  # exact neighbor indices and distances
                    gpu_config = faiss.GpuIndexIVFFlatConfig()
                    # Get physical GPU index with load balancing
                    gpu_config.device = _get_gpu_index()
                    index = faiss.GpuIndexIVFFlat(
                        gpu_resource,
                        target.shape[-1],
                        num_clusters,
                        faiss.METRIC_INNER_PRODUCT,
                        gpu_config,
                    )  # approximate neighbor indices and exact distances
                else:
                    # use CPU
                    # index = faiss.IndexFlatIP(target.shape[-1])  # exact neighbor indices and distances
                    index = faiss.IndexIVFFlat(
                        faiss.IndexFlatIP(target.shape[-1]),
                        target.shape[-1],
                        num_clusters,
                        faiss.METRIC_INNER_PRODUCT,
                    )  # approximate neighbor indices and exact distances

                index.train(target)
                index.add(target)

                # cache index
                index_cache[backend][index_hash] = index
            else:
                index = index_cache[backend][index_hash]

            dots, neighbor_indices = index.search(source, k=num_neighbors)

            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)

        # cache result
        result_cache[result_hash] = neighbor_indices, angles
    else:
        neighbor_indices, angles = result_cache[result_hash]

    return (to_numpy(neighbor_indices), to_numpy(angles))
