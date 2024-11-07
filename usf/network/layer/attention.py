#!/usr/bin/env python3
#
# Created on Wed Oct 29 2025 21:33:03
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


class Attention(nn.Module, ABC):
    """Abstract base class and factory for attention backends.

    ``Attention`` uses ``__new__`` as a factory: instantiation always returns
    a concrete backend subclass selected by the ``backend`` keyword argument
    (one of ``"xformers"``, ``"flex attention"``, ``"plain"``).

    **Subclass contract** — each backend must implement:

    - ``forward(query, key, value, attention_bias) -> torch.Tensor``: the
      attention computation with shapes ``(B, N, H, D)`` for Q/K/V and an
      optional bias tensor.
    """

    _BACKENDS = {
        "xformers": (".attention_backend.xformers", "XFormers"),
        "flex attention": (".attention_backend.flex_attention", "FlexAttention"),
        "plain": (".attention_backend.plain", "Plain"),
    }

    def __new__(cls, backend: str | None = None, *args, **kwargs) -> Attention:
        """Factory constructor — resolve ``backend`` to a concrete subclass.

        Args:
            backend (str | None): Backend identifier. Required.

        Returns:
            Attention: An instance of the concrete backend subclass.
        """
        assert backend is not None, "Attention requires keyword argument 'backend'"
        assert (
            backend in cls._BACKENDS
        ), f"Unsupported backend {backend}. Currently supports {cls._BACKENDS.keys()}"

        module_path, class_name = cls._BACKENDS[backend]
        module = import_module(module_path, package=__package__)
        subclass = getattr(module, class_name)
        subclass_instance = super().__new__(subclass)
        object.__setattr__(subclass_instance, "backend", backend)
        return subclass_instance

    def __init__(self, *args, **kwargs) -> None:
        """Store constructor positional/keyword args for pickle support."""
        super().__init__()
        self._init_args = args
        self._init_kwargs = kwargs

    def __getnewargs_ex__(self) -> tuple[tuple, dict]:
        """Support pickling by returning the original constructor arguments.

        Returns:
            tuple[tuple, dict]: ``(positional_args, keyword_args)`` for ``__new__``.
        """
        return (self.backend, *self._init_args), self._init_kwargs

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor | Any | None,
    ) -> torch.Tensor:
        """Attention implementation

        Args:
            query (torch.Tensor): shape (B, N_q, H, D)
            key (torch.Tensor): shape (B, N_kv, H, D)
            value (torch.Tensor): shape (B, N_kv, H, D_v)
            attention_bias (torch.Tensor | Any | None): shape (B, H, N_q, N_kv)

        Returns:
            torch.Tensor: shape (B, N_q, H, D_v)
        """
        raise NotImplementedError(
            "This method must be implemented in the derived class"
        )
