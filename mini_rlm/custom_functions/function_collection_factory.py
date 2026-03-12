from typing import List

from mini_rlm.custom_functions.data_model import (
    FunctionBase,
    FunctionCollection,
)
from mini_rlm.custom_functions.functions import (
    convert_pdf_page_to_image_data_function,
    convert_pdf_page_to_text_function,
    llm_image_query_factory,
    llm_pdf_query_factory,
    llm_query_factory,
    open_image_data_function,
    pdf_page_length_function,
    rlm_query_factory,
)


def minimal_function_collection() -> FunctionCollection:
    """Create a minimal FunctionCollection with LLM and recursive query functions."""
    return FunctionCollection(functions=[llm_query_factory, rlm_query_factory])


def image_function_collection() -> FunctionCollection:
    """Create a FunctionCollection with image-related functions and query functions."""
    functions: List[FunctionBase] = [
        open_image_data_function,
        llm_query_factory,
        llm_image_query_factory,
        rlm_query_factory,
    ]
    return FunctionCollection(functions=functions)


def pdf_function_collection() -> FunctionCollection:
    """Create a FunctionCollection with PDF-related functions and query functions."""
    functions: List[FunctionBase] = [
        convert_pdf_page_to_image_data_function,
        convert_pdf_page_to_text_function,
        pdf_page_length_function,
        llm_query_factory,
        llm_image_query_factory,
        llm_pdf_query_factory,
        rlm_query_factory,
    ]
    return FunctionCollection(functions=functions)
