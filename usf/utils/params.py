#!/usr/bin/env python3
#
# Created on Thu Jul 06 2023 15:46:52
# Author: Mukai (Tom Notch) Yu, Yao He
# Email: mukaiy@andrew.cmu.edu, yaohe@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute, the AirLab
#
# Copyright Ⓒ 2023 Mukai (Tom Notch) Yu, Yao He
#
"""Global constants and dtype lookup tables shared across the package."""

from __future__ import annotations

import numpy as np
import torch

SENSOR_TYPE = {
    0: "IMU",
    1: "Non-Depth Camera",
    2: "Depth Camera",
    3: "LiDAR",
}

NON_DEPTH_CAMERA_SENSING_MODALITY = {
    0: "RGB",
    1: "Thermal",
}

CAMERA_MODEL_TYPE = {
    0: "Pinhole",
    1: "Mei",  # Christopher Mei, first author of https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
}

# --- dtype maps (keep distinct purposes explicit; names are unique) ---

# Config / checkpoint: string name -> torch.dtype
DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}

# OpenCV YAML !!opencv-matrix: single-letter ``dt`` field -> NumPy dtype
# (see ``opencv_matrix_constructor`` in ``usf.utils.files``).
OPENCV_YAML_MATRIX_DT_MAP: dict[str, np.dtype] = {
    "u": np.dtype(np.uint8),
    "i": np.dtype(np.int32),
    "f": np.dtype(np.float32),
    "d": np.dtype(np.float64),
}

# Packed binary blobs (e.g. spherical web export): uint8 code -> NumPy dtype (multi-byte values little-endian on disk).
BINARY_DTYPE_CODE_TO_NUMPY: dict[int, np.dtype] = {
    1: np.dtype(np.float32),
    2: np.dtype(np.float64),
    3: np.dtype(np.uint8),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.int64),
    7: np.dtype(np.bool_),
}

NUMPY_DTYPE_TO_BINARY_CODE: dict[np.dtype, int] = {
    np.dtype(dtype): code for code, dtype in BINARY_DTYPE_CODE_TO_NUMPY.items()
}

# USF spherical ``.bin`` on-disk format (batch ``(B, N, C)``; ``SphericalImage`` uses ``B=1``).
# See encode/decode in ``usf.utils.spherical_image``.
SPHERICAL_BIN_MAGIC = b"USFBSIMG"
SPHERICAL_BIN_VERSION = 1
SPHERICAL_BIN_HEADER_SIZE = 64
