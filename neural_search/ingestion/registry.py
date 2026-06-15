"""Registry of ingestion adapters by source name.

Each entry maps a source name to a callable that returns a list of raw
records. Adapters are imported lazily to avoid unnecessary heavy imports.

Usage:
    from neural_search.ingestion.registry import ADAPTER_REGISTRY, run_adapter
    records = run_adapter("neurovault", limit=100)
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Registry maps source_name -> (fetch_function, default_kwargs)
# All fetch functions must accept a `limit: int` kwarg.
_REGISTRY: dict[str, tuple[Callable[..., list[dict[str, Any]]], dict[str, Any]]] = {}


def register(source_name: str, **default_kwargs: Any):
    """Decorator to register an adapter fetch function."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[source_name] = (fn, default_kwargs)
        return fn
    return decorator


def run_adapter(source_name: str, limit: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
    """Run a registered adapter and return raw records."""
    if source_name not in _REGISTRY:
        raise ValueError(f"Unknown adapter: {source_name}. Available: {list(_REGISTRY.keys())}")
    fn, defaults = _REGISTRY[source_name]
    merged = {**defaults, **kwargs, "limit": limit}
    return fn(**merged)


def list_adapters() -> list[str]:
    """Return names of all registered adapters."""
    return sorted(_REGISTRY.keys())
