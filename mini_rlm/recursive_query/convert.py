from pathlib import Path
from typing import Any

from mini_rlm.recursive_query.data_model import (
    RecursiveQueryConfig,
    RecursiveQueryRuntime,
)
from mini_rlm.repl_session import ReplSessionLimits


def default_recursive_query_config() -> RecursiveQueryConfig:
    return RecursiveQueryConfig()


def resolve_recursive_query_runtime(
    runtime: RecursiveQueryRuntime | None,
    config: RecursiveQueryConfig,
) -> RecursiveQueryRuntime:
    if runtime is not None:
        return runtime
    return RecursiveQueryRuntime(remaining_depth=config.max_depth)


def build_child_recursive_query_runtime(
    runtime: RecursiveQueryRuntime,
) -> RecursiveQueryRuntime:
    if runtime.remaining_depth <= 0:
        raise ValueError("rlm_query max_depth exceeded")
    return RecursiveQueryRuntime(remaining_depth=runtime.remaining_depth - 1)


def build_child_repl_limits(config: RecursiveQueryConfig) -> ReplSessionLimits:
    return ReplSessionLimits(
        token_limit=config.child_token_limit,
        iteration_limit=config.child_iteration_limit,
        timeout_seconds=config.child_timeout_seconds,
        error_threshold=config.child_error_threshold,
        compacting_threshold_rate=config.child_compacting_threshold_rate,
    )


def list_inherited_file_paths(
    parent_temp_dir: str,
    inherit_parent_files: bool,
) -> list[Path]:
    if not inherit_parent_files:
        return []
    parent_dir = Path(parent_temp_dir)
    if not parent_dir.exists():
        return []
    return sorted([path for path in parent_dir.iterdir() if path.is_file()])


def extract_inherited_context_payload(
    parent_locals: dict[str, Any],
) -> dict[str, Any] | list[Any] | str | None:
    context_payload = parent_locals.get("context_0")
    if isinstance(context_payload, (dict, list, str)):
        return context_payload
    return None
