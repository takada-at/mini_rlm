from typing import Any, Dict

from mini_rlm.custom_functions.data_model import FunctionCollection
from mini_rlm.custom_functions.functions import (
    convert_pdf_page_to_image_data_function,
    create_query_image_llm,
    create_query_llm,
    create_query_pdf_llm,
    pdf_page_length_function,
)
from mini_rlm.llm.context_factory import create_request_context
from mini_rlm.llm.data_model import Endpoint


def create_function_collection(
    endpoint: Endpoint, request_params: Dict[str, Any]
) -> FunctionCollection:
    """Create a FunctionCollection with predefined functions and LLM query functions."""
    request_context = create_request_context(endpoint, request_params)
    functions = [
        convert_pdf_page_to_image_data_function,
        pdf_page_length_function,
        create_query_llm(request_context),
        create_query_image_llm(request_context),
        create_query_pdf_llm(request_context),
    ]
    return FunctionCollection(functions=functions)
