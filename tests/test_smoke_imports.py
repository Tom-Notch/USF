#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Smoke tests: verify every USF module can be imported without error."""

import importlib

import pytest

MODULES = [
    # core
    "usf",
    # utils
    "usf.utils.cache",
    "usf.utils.debug",
    "usf.utils.files",
    "usf.utils.image",
    "usf.utils.lr_scheduler",
    "usf.utils.nearest_neighbor",
    "usf.utils.positional_encoding",
    "usf.utils.spherical",
    "usf.utils.spherical_image",
    "usf.utils.torch_numpy",
    # datasets
    "usf.dataset.iou",
    "usf.dataset.mnist",
    "usf.dataset.pandora",
    "usf.dataset.stanford2D3DS",
    # samplers
    "usf.sampler.sampler",
    "usf.sampler.location.location_sampler",
    "usf.sampler.location.icosahedron",
    "usf.sampler.location.fibonacci",
    "usf.sampler.location.healpix",
    "usf.sampler.location.equirectangular",
    "usf.sampler.value.value_sampler",
    "usf.sampler.value.nearest_neighbor",
    "usf.sampler.value.radial_basis_function",
    # network layers
    "usf.network.layer.attention",
    "usf.network.layer.spherical.generic_spherical_cnn",
    "usf.network.layer.spherical.activation",
    "usf.network.layer.spherical.batchnorm",
    "usf.network.layer.spherical.circle_pool",
    "usf.network.layer.spherical.global_pool",
    "usf.network.layer.spherical.sa",
    "usf.network.layer.spherical.backend.circle_cnn",
    "usf.network.layer.spherical.backend.ripple_cnn",
    "usf.network.layer.spherical.backend.spherical_cnn",
    "usf.network.layer.spherical.backend.wave_cnn",
    "usf.network.layer.spherical.backend.weighting_function.weighting_function",
    "usf.network.layer.planar.sa",
    # network blocks
    "usf.network.block.planar.cbna",
    "usf.network.block.planar.c3k",
    "usf.network.block.planar.c3k2",
    "usf.network.block.planar.yolov11",
    "usf.network.block.spherical.cbna",
    "usf.network.block.spherical.c3k",
    "usf.network.block.spherical.c3k2",
    "usf.network.block.spherical.yolov11",
    # models
    "usf.network.model.mnist",
    "usf.network.model.object_detection",
    "usf.network.model.semantic_segmentation",
    # visualization
    "usf.visualization.spherical_layer",
    "usf.visualization.spherical_projection",
    # CLI entry points
    "usf.train",
    "usf.evaluate",
    "usf.generate_lens_normal_map",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name: str) -> None:
    """Verify that a USF module can be imported without raising.

    Args:
        module_name (str): Fully qualified module name to import.
    """
    importlib.import_module(module_name)
