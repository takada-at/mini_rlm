from typing import Callable

from mini_rlm.custom_functions.data_model import Argument, Function, FunctionFactory
from mini_rlm.image import ImageData, open_image_data
from mini_rlm.llm import RequestContext, query_functions
from mini_rlm.pdf import convert_pdf_page_to_image_data, pdf_page_length

convert_pdf_page_to_image_data_function = Function(
    name="convert_pdf_page_to_image_data",
    description="Convert a PDF page to ImageData",
    arguments=[
        Argument(name="file_path", description="Path to the PDF file", type=str),
        Argument(
            name="page_index",
            description="Page number to convert (0-indexed)",
            type=int,
        ),
    ],
    function=convert_pdf_page_to_image_data,
    return_type=ImageData,
)

pdf_page_length_function = Function(
    name="pdf_page_length",
    description="Return the number of pages in a PDF file",
    arguments=[Argument(name="pdf_path", description="Path to the PDF file", type=str)],
    function=pdf_page_length,
    return_type=int,
)

open_image_data_function = Function(
    name="open_image_data",
    description="Open an image file and convert it to ImageData",
    arguments=[
        Argument(name="file_path", description="Path to the image file", type=str)
    ],
    function=open_image_data,
    return_type=ImageData,
)


def create_query_llm(request_context: RequestContext) -> Callable[[str], str]:
    def query_llm(text: str) -> str:
        return query_functions.text_query(request_context, text)

    return query_llm


query_llm_factory = FunctionFactory(
    name="query_llm",
    description="Query the LLM with text input",
    arguments=[Argument(name="text", description="Text input for the LLM", type=str)],
    return_type=str,
    factory=create_query_llm,
)


def create_query_image_llm(
    request_context: RequestContext,
) -> Callable[[str, ImageData], str]:
    def query_image_llm(text: str, image_data: ImageData) -> str:
        return query_functions.image_query(request_context, text, image_data)

    return query_image_llm


query_image_llm_factory = FunctionFactory(
    name="query_image_llm",
    description="Query the LLM with text and image input",
    arguments=[
        Argument(name="text", description="Text input for the LLM", type=str),
        Argument(
            name="image_data",
            description="Image data input for the LLM",
            type=ImageData,
        ),
    ],
    factory=create_query_image_llm,
    return_type=str,
)


query_image_llm_factory = FunctionFactory(
    name="query_image_llm",
    description="Query the LLM with text and image input",
    arguments=[
        Argument(name="text", description="Text input for the LLM", type=str),
        Argument(
            name="image_data",
            description="Image data input for the LLM",
            type=ImageData,
        ),
    ],
    return_type=str,
    factory=create_query_image_llm,
)


def create_query_pdf_llm(
    request_context: RequestContext,
) -> Callable[[str, str, int], str]:
    def query_pdf_llm(text: str, pdf_path: str, page_index: int) -> str:
        image_data = convert_pdf_page_to_image_data(pdf_path, page_index)
        return query_functions.image_query(request_context, text, image_data)

    return query_pdf_llm


query_pdf_llm_factory = FunctionFactory(
    name="query_pdf_llm",
    description="Query the LLM with text and PDF input",
    arguments=[
        Argument(name="text", description="Text input for the LLM", type=str),
        Argument(
            name="pdf_path",
            description="Path to the PDF file for the LLM",
            type=str,
        ),
        Argument(
            name="page_index",
            description="Page number to query (0-indexed)",
            type=int,
        ),
    ],
    factory=create_query_pdf_llm,
    return_type=str,
)


query_pdf_llm_factory = FunctionFactory(
    name="query_pdf_llm",
    description="Query the LLM with text and PDF input",
    arguments=[
        Argument(name="text", description="Text input for the LLM", type=str),
        Argument(
            name="pdf_path", description="Path to the PDF file for the LLM", type=str
        ),
        Argument(
            name="page_index", description="Page number to query (0-indexed)", type=int
        ),
    ],
    return_type=str,
    factory=create_query_pdf_llm,
)
