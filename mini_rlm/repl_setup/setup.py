from pathlib import Path
from typing import Any, Dict, List

from mini_rlm.custom_functions import (
    Function,
    FunctionCollection,
    FunctionFactory,
    image_function_collection,
)
from mini_rlm.llm.data_model import RequestContext
from mini_rlm.repl import add_file, add_function, create_repl
from mini_rlm.repl_setup.data_model import ReplContext


def setup_repl(
    request_context: RequestContext,
    setup_code: str | None = None,
    context_payload: Dict[str, Any] | None = None,
    file_pathes: List[Path] | None = None,
    functions: FunctionCollection | None = None,
) -> ReplContext:
    """Create and initialise a new REPL instance."""
    state = create_repl(setup_code, context_payload)
    if file_pathes:
        for file_path in file_pathes:
            if not file_path.is_file():
                raise FileNotFoundError(f"File not found: {file_path}")
            with file_path.open("rb") as f:
                add_file(state, file_path.name, f)
    if functions is None:
        functions = image_function_collection()
    for func in functions.functions:
        if isinstance(func, FunctionFactory):
            pyfunc = func.factory(request_context)
        else:
            assert isinstance(func, Function)
            pyfunc = func.function
        add_function(state, func.name, pyfunc)
    return ReplContext(
        request_context=request_context,
        repl_state=state,
        functions=functions,
    )
