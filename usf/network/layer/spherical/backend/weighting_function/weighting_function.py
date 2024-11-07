#!/usr/bin/env python3
#
# Created on Sun Mar 16 2025 19:55:50
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2025 Mukai (Tom Notch) Yu
#
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Any

import torch
import torch.nn as nn


class WeightingFunction(nn.Module, ABC):
    """Abstract base class and factory for spherical conv weighting functions.

    ``WeightingFunction`` uses ``__new__`` as a factory: instantiation returns
    a concrete subclass (``Continuous`` or ``Discrete``) selected by
    ``config["function"]``.

    **Subclass contract** — each backend must implement:

    - ``forward(x) -> torch.Tensor``: map input features of shape
      ``(..., in_channels)`` to weights of shape ``(..., num_functions, out_channels)``.
    """

    _WEIGHTING_FUNCTION_TYPES = {
        "continuous": (".continuous", "Continuous"),
        "discrete": (".discrete", "Discrete"),
    }

    def __new__(
        cls,
        config: dict[str, Any],
        *args,
        **kwargs,
    ) -> WeightingFunction:
        """Factory constructor — resolve ``config["function"]`` to a concrete subclass.

        Args:
            config (dict[str, Any]): Must contain ``"function"`` (str).

        Returns:
            WeightingFunction: An instance of the selected concrete subclass.
        """
        function_type = config.get("function", None)
        assert function_type is not None, "Key function must be provided in config dict"

        weighting_function_class_config = cls._WEIGHTING_FUNCTION_TYPES.get(
            function_type, None
        )
        assert (
            weighting_function_class_config is not None
        ), f"unsupported weighting function type {function_type}, supported types: {cls._WEIGHTING_FUNCTION_TYPES.keys()}"

        module_path, class_name = weighting_function_class_config

        # Import the module and get the class. This is done lazily
        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        subclass_instance.__init__(config, *args, **kwargs)
        return subclass_instance

    def __init__(
        self,
        config: dict[str, Any],
        *args,
        **kwargs,
    ) -> None:
        """Store config and extract shared attributes.

        Args:
            config (dict[str, Any]): Must contain ``in_channels``, ``out_channels``,
                ``num_functions``.
        """
        super().__init__()
        self.config = config

        # internalize common attributes
        self.in_channels = self.config["in_channels"]
        self.out_channels = self.config["out_channels"]
        self.num_functions = self.config["num_functions"]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): shape (..., in_channels)

        Raises:
            NotImplementedError: must be implemented by child class

        Returns:
            torch.Tensor: shape (..., num_functions, out_channels)
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )


class MultiBranchWeightingFunction(nn.Module):
    """
    A multi-branch WeightingFunction that processes multiple input in parallel.

    The input is provided as a dictionary mapping branch names to tensors of shape (..., in_channels).
    For each branch, a corresponding WeightingFunction is applied (with its own configuration) so that the output is
    of shape (..., num_functions).
    """

    def __init__(self, branch_configs: dict[str, dict[str, Any]]):
        """Initialize MultiBranchWeightingFunction.

        Args:
            branch_configs (dict[str, dict[str, Any]]): A dict mapping branch names (e.g. 'geodesic', 'direction') to their configuration,
                where each configuration is itself a dict with kwargs for the WeightingFunction constructor.
        """
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                branch_name: WeightingFunction(config)
                for branch_name, config in branch_configs.items()
            }
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            x (dict[str, torch.Tensor]): A dictionary mapping branch names to their input tensors.
               Each tensor should have shape (..., in_channels) as required by the corresponding branch.

        Returns:
            dict[srt, torch.Tensor]: A dictionary mapping branch names to their output tensors, each tensor should have shape (..., num_functions, out_channels)
        """
        futures = {
            branch_name: torch.jit.fork(function.forward, x[branch_name])
            for branch_name, function in self.branches.items()
        }
        return {
            branch_name: torch.jit.wait(future)
            for branch_name, future in futures.items()
        }
