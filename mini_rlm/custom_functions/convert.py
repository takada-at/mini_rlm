from textwrap import dedent
from typing import Any, Callable

from mini_rlm.custom_functions.data_model import (
    Argument,
    Function,
    FunctionBase,
    FunctionCollection,
)
from mini_rlm.recursive_query import RecursiveQueryRuntime


def convert_to_function(func: Callable[..., Any]) -> Function:
    """
    Convert a Python function to a Function object.

    Args:
        func (Callable[..., Any]): The Python function to convert.

    Returns:
        Function: The converted Function object.
    """
    name = func.__name__
    description = func.__doc__ or ""
    arguments = []
    for param in func.__annotations__.items():
        arg_name, arg_type = param
        arguments.append(Argument(name=arg_name, description="", type=arg_type))
    return_type = func.__annotations__.get("return", None)
    return Function(
        name=name,
        description=description,
        arguments=arguments,
        return_type=return_type,
        function=func,
    )


def convert_function_to_string(func: FunctionBase) -> str:
    """
    Convert a Function object to a string representation.

    Args:
        func (FunctionBase): The Function object to convert.

    Returns:
        str: The string representation of the Function object.
    """
    args_str = ", ".join([f"{arg.name}: {arg.type.__name__}" for arg in func.arguments])

    doc_lines = []
    if func.description:
        doc_lines.extend(dedent(func.description).strip("\n").splitlines())

    arg_desc_lines = [
        f"{arg.name} ({arg.type.__name__}): {arg.description}"
        for arg in func.arguments
        if arg.description
    ]
    if arg_desc_lines:
        if doc_lines:
            doc_lines.append("")
        doc_lines.append("Args:")
        doc_lines.extend([f"    {line}" for line in arg_desc_lines])

    if doc_lines:
        docstring = '"""' + "\n" + "\n".join(doc_lines) + '\n"""'
        indented_docstring = ["    " + line for line in docstring.splitlines()]
    else:
        indented_docstring = []
    ret_type_str = func.return_type.__name__ if func.return_type else "None"
    return f"def {func.name}({args_str}) -> {ret_type_str}:\n{'\n'.join(indented_docstring)}\n...\n"


def convert_function_collection_to_string(func_collection: FunctionCollection) -> str:
    """
    Convert a FunctionCollection object to a string representation.

    Args:
        func_collection (FunctionCollection): The FunctionCollection object to convert.

    Returns:
        str: The string representation of the FunctionCollection object.
    """
    return "\n".join(
        [
            f"```\n{convert_function_to_string(func)}\n```"
            for func in func_collection.functions
        ]
    )


def filter_function_collection_for_runtime(
    function_collection: FunctionCollection,
    runtime: RecursiveQueryRuntime | None,
) -> FunctionCollection:
    if runtime is None or runtime.remaining_depth > 0:
        return FunctionCollection(functions=list(function_collection.functions))
    return FunctionCollection(
        functions=[
            function
            for function in function_collection.functions
            if function.name != "rlm_query"
        ]
    )


def merge_function_collections(
    *function_collections: FunctionCollection,
) -> FunctionCollection:
    merged_functions: list[FunctionBase] = []
    seen_names: set[str] = set()
    for function_collection in function_collections:
        for function in function_collection.functions:
            if function.name in seen_names:
                continue
            seen_names.add(function.name)
            merged_functions.append(function)
    return FunctionCollection(functions=merged_functions)
