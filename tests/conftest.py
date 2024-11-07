#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Shared pytest fixtures for USF tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def rng_np() -> np.random.Generator:
    """Seeded numpy random generator for reproducible tests.

    Returns:
        np.random.Generator: Deterministic RNG seeded at 42.
    """
    return np.random.default_rng(42)


@pytest.fixture
def device() -> torch.device:
    """Always CPU for CI tests.

    Returns:
        torch.device: CPU device.
    """
    return torch.device("cpu")
