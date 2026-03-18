from pathlib import Path
from typing import Any, Dict, List

from mini_rlm.custom_functions import (
    Function,
    FunctionCollection,
    FunctionFactory,
    FunctionFactoryContext,
    filter_function_collection_for_runtime,
    image_function_collection,
)
from mini_rlm.debug_logger import get_logger
from mini_rlm.llm import RequestContext
from mini_rlm.recursive_query import RecursiveQueryRuntime
from mini_rlm.repl import add_file, add_function, create_repl
from mini_rlm.repl_setup.data_model import ReplContext, ReplFileRef


def _iter_repl_files(
    file_pathes: List[Path] | None,
    files: List[ReplFileRef] | None,
) -> list[tuple[Path, str]]:
    repl_files: list[tuple[Path, str]] = []
    for file_ref in files or []:
        target_name = file_ref.target_name or file_ref.source_path.name
        repl_files.append((file_ref.source_path, target_name))
    for file_path in file_pathes or []:
        repl_files.append((file_path, file_path.name))
    return repl_files


def setup_repl(
    request_context: RequestContext,
    setup_code: str | None = None,
    context_payload: Dict[str, Any] | list[Any] | str | None = None,
    file_pathes: List[Path] | None = None,
    files: List[ReplFileRef] | None = None,
    functions: FunctionCollection | None = None,
    recursive_query_runtime: RecursiveQueryRuntime | None = None,
) -> ReplContext:
    """Create and initialise a new REPL instance."""
    state = create_repl(setup_code, context_payload)
    seen_target_names: set[str] = set()
    for source_path, target_name in _iter_repl_files(file_pathes, files):
        if not source_path.is_file():
            raise FileNotFoundError(f"File not found: {source_path}")
        if target_name in seen_target_names:
            raise ValueError(f"Duplicate REPL target name: {target_name}")
        seen_target_names.add(target_name)
        with source_path.open("rb") as f:
            add_file(state, target_name, f)
    if functions is None:
        functions = image_function_collection()
    functions = filter_function_collection_for_runtime(
        functions, recursive_query_runtime
    )
    logger = get_logger()
    for func in functions.functions:
        if isinstance(func, FunctionFactory):
            pyfunc = func.factory(
                FunctionFactoryContext(
                    request_context=request_context,
                    repl_state=state,
                    function_collection=functions,
                    recursive_query_runtime=recursive_query_runtime,
                )
            )
        else:
            assert isinstance(func, Function)
            pyfunc = func.function
        assert callable(pyfunc), f"Function {func.name} is not callable"
        add_function(state, func.name, pyfunc)
        logger.debug(f"Added function to REPL: {func.name}")
    return ReplContext(
        request_context=request_context,
        repl_state=state,
        functions=functions,
        recursive_query_runtime=recursive_query_runtime,
    )
