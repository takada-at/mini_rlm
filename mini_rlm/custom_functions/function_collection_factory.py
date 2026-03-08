from typing import List

from mini_rlm.custom_functions.data_model import (
    FunctionBase,
    FunctionCollection,
)
from mini_rlm.custom_functions.functions import (
    convert_pdf_page_to_image_data_function,
    convert_pdf_page_to_text_function,
    open_image_data_function,
    pdf_page_length_function,
    query_image_llm_factory,
    query_llm_factory,
    query_pdf_llm_factory,
)


def minimal_function_collection() -> FunctionCollection:
    """Create a minimal FunctionCollection with only the query_llm function."""
    return FunctionCollection(functions=[query_llm_factory])


def image_function_collection() -> FunctionCollection:
    """Create a FunctionCollection with image-related functions and LLM query functions."""
    functions: List[FunctionBase] = [
        open_image_data_function,
        query_llm_factory,
        query_image_llm_factory,
    ]
    return FunctionCollection(functions=functions)


def pdf_function_collection() -> FunctionCollection:
    """Create a FunctionCollection with PDF-related functions and LLM query functions."""
    functions: List[FunctionBase] = [
        convert_pdf_page_to_image_data_function,
        convert_pdf_page_to_text_function,
        pdf_page_length_function,
        query_llm_factory,
        query_image_llm_factory,
        query_pdf_llm_factory,
    ]
    return FunctionCollection(functions=functions)
