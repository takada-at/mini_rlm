import copy
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import IO, Any, Dict

from mini_rlm.repl.data_model import ReplResult, ReplState
from mini_rlm.repl.executor import execute_repl_execution

# =============================================================================
# Safe Builtins
# =============================================================================

# Blocks dangerous operations like eval/exec/input
SAFE_BUILTINS: dict[str, Any] = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


# =============================================================================
# REPL lifecycle
# =============================================================================


def create_repl(
    setup_code: str | None = None,
    context_payload: Dict[str, Any] | None = None,
) -> ReplState:
    """Create and initialise a new REPL state."""
    temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
    state = ReplState(temp_dir=temp_dir)

    # Closures so the helpers always operate on the live state object
    def _final_var(variable_name: str | Any) -> str:
        return final_var(state, variable_name)

    def _final(text: str) -> str:
        return final(state, text)

    def _show_vars() -> str:
        return show_vars(state)

    state.globals = {
        "__builtins__": SAFE_BUILTINS.copy(),
        "__name__": "__main__",
        "FINAL_VAR": _final_var,
        "FINAL": _final,
        "SHOW_VARS": _show_vars,
    }
    state.locals = {}
    state.reserved_globals = {
        "FINAL_VAR": _final_var,
        "SHOW_VARS": _show_vars,
    }

    if context_payload is not None:
        load_context(state, context_payload)

    if setup_code:
        execute_code(state, setup_code)

    return state


def cleanup(state: ReplState) -> None:
    """Remove the temp directory and clear the namespace."""
    try:
        shutil.rmtree(state.temp_dir)
    except Exception:
        pass
    state.globals.clear()
    state.locals.clear()


# =============================================================================
# Namespace helpers
# =============================================================================


def add_function(state: ReplState, name: str, fn: Any) -> None:
    """Add a callable to the REPL global namespace.

    The function is also registered as a reserved global so it survives
    re-execution without being overwritten by user code.
    """
    state.globals[name] = fn
    state.reserved_globals[name] = fn


# =============================================================================
# Built-in REPL helpers (exposed as FINAL_VAR / SHOW_VARS inside the sandbox)
# =============================================================================


def final(state: ReplState, text: str) -> str:
    """Return a string value as the final answer, or stringify a direct value."""
    answer = str(text)
    state.last_final_answer = answer
    return answer


def final_var(state: ReplState, variable_name: str | Any) -> str:
    """Return a variable's value as the final answer, or stringify a direct value."""
    if not isinstance(variable_name, str):
        answer = str(variable_name)
        state.last_final_answer = answer
        return answer

    variable_name = variable_name.strip().strip("\"'")
    if variable_name in state.locals:
        answer = str(state.locals[variable_name])
        state.last_final_answer = answer
        return answer

    available = [k for k in state.locals if not k.startswith("_")]
    if available:
        return (
            f"Error: Variable '{variable_name}' not found. "
            f"Available variables: {available}. "
            f"You must create and assign a variable BEFORE calling FINAL_VAR on it."
        )
    return (
        f"Error: Variable '{variable_name}' not found. "
        f"No variables have been created yet. "
        f"You must create and assign a variable in a REPL block BEFORE calling FINAL_VAR on it."
    )


def show_vars(state: ReplState) -> str:
    """Show all user-defined variables in the sandbox."""
    available = {
        k: type(v).__name__ for k, v in state.locals.items() if not k.startswith("_")
    }
    if not available:
        return "No variables created yet."
    return f"Available variables: {available}"


def execute_code(state: ReplState, code: str) -> ReplResult:
    """Execute *code* in the persistent sandbox and return a ReplResult."""
    return execute_repl_execution(state, code)


# =============================================================================
# Context and history management
# =============================================================================


def add_context(
    state: ReplState,
    context_payload: Dict[str, Any] | list | str | None,
    context_index: int | None = None,
) -> int:
    """Load *context_payload* into the sandbox as ``context_<index>``."""
    if context_index is None:
        context_index = state.context_count

    var_name = f"context_{context_index}"

    context_path = os.path.join(state.temp_dir, f"context_{context_index}.json")
    with open(context_path, "w") as f:
        json.dump(context_payload, f)
    execute_code(
        state,
        f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)",
    )

    if context_index == 0:
        execute_code(state, f"context = {var_name}")

    state.context_count = max(state.context_count, context_index + 1)
    return context_index


def load_context(state: ReplState, context_payload: Dict[str, Any] | None) -> None:
    """Shorthand: load *context_payload* as ``context_0`` / ``context``."""
    add_context(state, context_payload, 0)


def add_history(
    state: ReplState,
    message_history: list[Dict[str, Any]],
    history_index: int | None = None,
) -> int:
    """Store *message_history* as ``history_<index>`` in the sandbox locals."""
    if history_index is None:
        history_index = state.history_count

    var_name = f"history_{history_index}"
    state.locals[var_name] = copy.deepcopy(message_history)

    if history_index == 0:
        state.locals["history"] = state.locals[var_name]

    state.history_count = max(state.history_count, history_index + 1)
    return history_index


def add_file(state: ReplState, filename: str, source_file: IO[bytes]) -> str:
    """Add a file with *filename* and *content* to the REPL temp directory."""
    safe_name = Path(filename).name  # Prevent directory traversal
    file_path = Path(state.temp_dir) / safe_name
    with file_path.open("wb") as f:
        f.write(source_file.read())
    return safe_name
