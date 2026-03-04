from textwrap import dedent

from mini_rlm.custom_functions.data_model import Function, FunctionCollection


def convert_to_function(func: callable) -> Function:
    """
    Convert a Python function to a Function object.

    Args:
        func (callable): The Python function to convert.

    Returns:
        Function: The converted Function object.
    """
    name = func.__name__
    description = func.__doc__ or ""
    arguments = []
    for param in func.__annotations__.items():
        arg_name, arg_type = param
        arguments.append({"name": arg_name, "description": "", "type": arg_type})
    return Function(name=name, description=description, arguments=arguments)


def convert_function_to_string(func: Function) -> str:
    """
    Convert a Function object to a string representation.

    Args:
        func (Function): The Function object to convert.

    Returns:
        str: The string representation of the Function object.
    """
    args_str = ", ".join([f"{arg.name}: {arg.type.__name__}" for arg in func.arguments])
    if func.description:
        docstring = '"""' + "\n" + func.description + '\n"""'
        indented_docstring = ["    " + line for line in dedent(docstring).splitlines()]
    else:
        indented_docstring = []
    ret_type_str = func.return_type.__name__ if func.return_type else "None"
    return f"def {func.name}({args_str}) -> {ret_type_str}:\n{'\n'.join(indented_docstring)}\n\n"


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
