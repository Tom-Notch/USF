#!/usr/bin/env python3
#
# Created on Sun Apr 13 2025 2:24:04
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
from __future__ import annotations

import contextlib
from collections.abc import Generator
from copy import deepcopy

from cachetools import Cache


class CacheMode:
    """Thread-global singleton that controls whether LFU caches are active.

    The singleton maintains a boolean stack so that nested scopes (e.g.
    ``ephemeral_cache``) can temporarily disable caching and restore the
    prior state on exit.  All ``Cache`` instances registered via
    ``register_cache`` are tracked for snapshot/restore during ephemeral
    scopes.

    Use the module-level helpers (``is_cache_enabled``, ``set_cache_enabled``,
    ``register_cache``, ``ephemeral_cache``) rather than accessing the
    singleton directly.
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> CacheMode:
        """Return the singleton ``CacheMode`` instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the cache stack and registry (skipped on re-entry)."""

        # Prevent re-initialization
        if getattr(self, "_initialized", False):
            return

        self._cache_stack = [True]
        self._caches: dict[str, Cache] = {}
        self._initialized = True

    def is_enabled(self) -> bool:
        """Return whether caching is currently enabled (top of the stack).

        Returns:
            bool: True if caching is active.
        """
        return self._cache_stack[-1]

    def cache_stack(self) -> list[bool]:
        """Return a deep copy of the current cache-enabled stack.

        Returns:
            list[bool]: Snapshot of the boolean stack (bottom → top).
        """
        return deepcopy(self._cache_stack)

    def set_enabled(self, mode: bool) -> None:
        """Overwrite the current (top-of-stack) caching mode.

        Args:
            mode (bool): New enabled state.
        """
        self._cache_stack[-1] = mode

    def push(self, mode: bool) -> None:
        """Push a new caching mode onto the stack.

        Args:
            mode (bool): Enabled state for the new scope.
        """
        self._cache_stack.append(mode)

    def pop(self) -> None:
        """Pop the top caching mode, restoring the previous scope."""
        self._cache_stack.pop()

    def register_cache(self, name: str, cache: Cache) -> None:
        """Register a named ``Cache`` instance for snapshot/restore tracking.

        Args:
            name (str): Unique identifier for this cache.
            cache (Cache): The ``cachetools.Cache`` instance.
        """
        self._caches[name] = cache

    def snapshot_cache(self) -> dict[str, list[str]]:
        """Snapshot the cache state.

        Returns:
            dict[str, list[str]]: The snapshot of the cache state with cache key as value.
        """
        return {name: list(cache.keys()) for name, cache in self._caches.items()}

    def restore_cache(self, snapshot: dict[str, list[str]]) -> None:
        """Restore the cache state from the snapshot.

        Evicts any keys added since the snapshot was taken while keeping
        pre-existing entries intact.

        Args:
            snapshot (dict[str, list[str]]): The snapshot returned by ``snapshot_cache``.
        """
        for name, cache in self._caches.items():
            if name not in snapshot:
                continue
            for key in list(cache.keys()):
                if key not in snapshot[name]:
                    cache.pop(key)


cache_mode = CacheMode()


def is_cache_enabled() -> bool:
    """Return whether LFU caching is currently enabled.

    Returns:
        bool: True if caching is active (top of the cache-mode stack).
    """
    return cache_mode.is_enabled()


def get_cache_stack() -> list[bool]:
    """Return a snapshot of the cache-enabled stack.

    Returns:
        list[bool]: Deep copy of the boolean stack (bottom → top).
    """
    return cache_mode.cache_stack()


def set_cache_enabled(mode: bool) -> None:
    """Overwrite the current caching mode at the top of the stack.

    Args:
        mode (bool): New enabled state.
    """
    cache_mode.set_enabled(mode)


def register_cache(name: str, cache: Cache) -> None:
    """Register a named ``Cache`` instance for ephemeral-scope tracking.

    Args:
        name (str): Unique identifier for this cache.
        cache (Cache): The ``cachetools.Cache`` instance to track.
    """
    cache_mode.register_cache(name, cache)


@contextlib.contextmanager
def ephemeral_cache(ephemeral: bool = True) -> Generator[None, None, None]:
    """Context manager for ephemeral caching — cache entries created in this block are cleaned up on exit.

    Args:
        ephemeral (bool, optional): If True, disable caching and clean up on exit. Defaults to True.

    Example:
        >>> with ephemeral_cache():
        ...     # All caches accumulated in this block will be evicted on exit
        ...     result = sampler(batch_spherical_image)
    """
    cache_mode.push((not ephemeral) and cache_mode.is_enabled())

    if not is_cache_enabled():
        snapshot = cache_mode.snapshot_cache()

    try:
        yield

    finally:
        if not is_cache_enabled():
            cache_mode.restore_cache(snapshot)

        cache_mode.pop()
