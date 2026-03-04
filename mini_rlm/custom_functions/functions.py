from mini_rlm.custom_functions.data_model import Function
from mini_rlm.image.data_model import ImageData
from mini_rlm.llm import query_functions
from mini_rlm.llm.data_model import RequestContext
from mini_rlm.pdf.convert import convert_pdf_page_to_image_data
from mini_rlm.pdf.pdf_util import pdf_page_length

convert_pdf_page_to_image_data_function = Function(
    name="convert_pdf_page_to_image_data",
    description="Convert a PDF page to ImageData",
    arguments=[
        {"name": "file_path", "description": "Path to the PDF file", "type": str},
        {
            "name": "page_index",
            "description": "Page number to convert (0-indexed)",
            "type": int,
        },
    ],
    function=convert_pdf_page_to_image_data,
    return_type=ImageData,
)

pdf_page_length_function = Function(
    name="pdf_page_length",
    description="Return the number of pages in a PDF file",
    arguments=[
        {"name": "pdf_path", "description": "Path to the PDF file", "type": str}
    ],
    function=pdf_page_length,
    return_type=int,
)


def create_query_llm(request_context: RequestContext) -> Function:
    def query_llm(text: str) -> str:
        return query_functions.text_query(request_context, text)

    return Function(
        name="query_llm",
        description="Query the LLM with text input",
        arguments=[
            {"name": "text", "description": "Text input for the LLM", "type": str}
        ],
        function=query_llm,
        return_type=str,
    )


def create_query_image_llm(request_context: RequestContext) -> Function:
    def query_image_llm(text: str, image_data: ImageData) -> str:
        return query_functions.image_query(request_context, text, image_data)

    return Function(
        name="query_image_llm",
        description="Query the LLM with text and image input",
        arguments=[
            {"name": "text", "description": "Text input for the LLM", "type": str},
            {
                "name": "image_data",
                "description": "Image data input for the LLM",
                "type": ImageData,
            },
        ],
        function=query_image_llm,
        return_type=str,
    )


def create_query_pdf_llm(request_context: RequestContext) -> Function:
    def query_pdf_llm(text: str, pdf_path: str, page_index: int) -> str:
        image_data = convert_pdf_page_to_image_data(pdf_path, page_index)
        return query_functions.image_query(request_context, text, image_data)

    return Function(
        name="query_pdf_llm",
        description="Query the LLM with text and PDF input",
        arguments=[
            {"name": "text", "description": "Text input for the LLM", "type": str},
            {
                "name": "pdf_path",
                "description": "Path to the PDF file for the LLM",
                "type": str,
            },
            {
                "name": "page_index",
                "description": "Page number to query (0-indexed)",
                "type": int,
            },
        ],
        function=query_pdf_llm,
        return_type=str,
    )
