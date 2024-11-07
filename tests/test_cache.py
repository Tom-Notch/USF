#!/usr/bin/env python3
#
# Created on Sun Apr 05 2026 14:40:54
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2026 Mukai (Tom Notch) Yu
#
"""Unit tests for usf.utils.cache (CacheMode singleton, ephemeral_cache)."""

from cachetools import LFUCache

from usf.utils.cache import (
    CacheMode,
    ephemeral_cache,
    get_cache_stack,
    is_cache_enabled,
    register_cache,
    set_cache_enabled,
)


class TestCacheModeSingleton:
    """Test CacheMode singleton behavior."""

    def test_singleton(self) -> None:
        """CacheMode() should always return the same instance."""
        a = CacheMode()
        b = CacheMode()
        assert a is b

    def test_default_enabled(self) -> None:
        """Cache should be enabled by default."""
        assert is_cache_enabled()

    def test_set_and_restore(self) -> None:
        """set_cache_enabled should toggle the current level."""
        original = is_cache_enabled()
        set_cache_enabled(False)
        assert not is_cache_enabled()
        set_cache_enabled(original)
        assert is_cache_enabled() == original


class TestCacheStack:
    """Test push/pop stack semantics."""

    def test_push_pop(self) -> None:
        """Push/pop should nest correctly and restore depth."""
        cm = CacheMode()
        initial_depth = len(get_cache_stack())

        cm.push(False)
        assert not is_cache_enabled()
        assert len(get_cache_stack()) == initial_depth + 1

        cm.push(True)
        assert is_cache_enabled()
        assert len(get_cache_stack()) == initial_depth + 2

        cm.pop()
        assert not is_cache_enabled()
        cm.pop()
        assert len(get_cache_stack()) == initial_depth

    def test_nested_ephemeral_cache(self) -> None:
        """Nested ephemeral_cache should restore stack depth on exit."""
        initial_depth = len(get_cache_stack())
        with ephemeral_cache():
            assert len(get_cache_stack()) == initial_depth + 1
            with ephemeral_cache():
                assert len(get_cache_stack()) == initial_depth + 2
            assert len(get_cache_stack()) == initial_depth + 1
        assert len(get_cache_stack()) == initial_depth


class TestEphemeralCache:
    """Test ephemeral_cache context manager eviction and preservation."""

    def test_entries_evicted_on_exit(self) -> None:
        """Keys added inside ephemeral_cache should be removed on exit."""
        cache = LFUCache(maxsize=100)
        register_cache("test_ephemeral", cache)

        cache["pre_existing"] = 1

        with ephemeral_cache():
            cache["ephemeral_key"] = 2
            assert "ephemeral_key" in cache

        assert "ephemeral_key" not in cache
        assert "pre_existing" in cache

        # cleanup
        cache.clear()

    def test_ephemeral_false_keeps_entries(self) -> None:
        """ephemeral=False should behave like normal caching."""
        cache = LFUCache(maxsize=100)
        register_cache("test_non_ephemeral", cache)

        with ephemeral_cache(ephemeral=False):
            cache["kept_key"] = 42

        assert "kept_key" in cache

        # cleanup
        cache.clear()

    def test_restores_on_exception(self) -> None:
        """Cache state should be restored even if the block raises."""
        cache = LFUCache(maxsize=100)
        register_cache("test_exception", cache)

        cache["safe"] = 1

        try:
            with ephemeral_cache():
                cache["doomed"] = 2
                raise ValueError("intentional")
        except ValueError:
            pass

        assert "doomed" not in cache
        assert "safe" in cache

        # cleanup
        cache.clear()


class TestSnapshotRestore:
    """Test snapshot_cache / restore_cache behavior."""

    def test_snapshot_captures_keys(self) -> None:
        """Snapshot should contain all current cache keys."""
        cm = CacheMode()
        cache = LFUCache(maxsize=100)
        register_cache("test_snapshot", cache)

        cache["a"] = 1
        cache["b"] = 2
        snap = cm.snapshot_cache()

        assert "a" in snap["test_snapshot"]
        assert "b" in snap["test_snapshot"]

        # cleanup
        cache.clear()

    def test_restore_evicts_new_keys(self) -> None:
        """restore_cache should evict keys added after the snapshot."""
        cm = CacheMode()
        cache = LFUCache(maxsize=100)
        register_cache("test_restore", cache)

        cache["original"] = 1
        snap = cm.snapshot_cache()

        cache["new"] = 2
        cm.restore_cache(snap)

        assert "original" in cache
        assert "new" not in cache

        # cleanup
        cache.clear()
