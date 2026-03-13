"""Recursive query domain package."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "RecursiveQueryConfig": (
        "mini_rlm.recursive_query.data_model",
        "RecursiveQueryConfig",
    ),
    "RecursiveQueryRuntime": (
        "mini_rlm.recursive_query.data_model",
        "RecursiveQueryRuntime",
    ),
    "RecursiveQueryRequest": (
        "mini_rlm.recursive_query.data_model",
        "RecursiveQueryRequest",
    ),
    "RecursiveQueryResult": (
        "mini_rlm.recursive_query.data_model",
        "RecursiveQueryResult",
    ),
    "default_recursive_query_config": (
        "mini_rlm.recursive_query.convert",
        "default_recursive_query_config",
    ),
    "execute_recursive_query": (
        "mini_rlm.recursive_query.executor",
        "execute_recursive_query",
    ),
}

__all__ = [
    "RecursiveQueryConfig",
    "RecursiveQueryRuntime",
    "RecursiveQueryRequest",
    "RecursiveQueryResult",
    "default_recursive_query_config",
    "execute_recursive_query",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
