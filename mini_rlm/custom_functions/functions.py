from typing import Callable

import mini_rlm.recursive_query as recursive_query
from mini_rlm.custom_functions.data_model import (
    Argument,
    Function,
    FunctionFactory,
    FunctionFactoryContext,
)
from mini_rlm.image import ImageData, open_image_data
from mini_rlm.llm import image_query_with_usage, text_query_with_usage
from mini_rlm.pdf import (
    convert_pdf_page_to_image_data,
    convert_pdf_page_to_text,
    pdf_page_length,
)

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
convert_pdf_page_to_text_function = Function(
    name="convert_pdf_page_to_text",
    description="Convert a PDF page to text",
    arguments=[
        Argument(name="file_path", description="Path to the PDF file", type=str),
        Argument(
            name="page_index",
            description="Page number to convert (0-indexed)",
            type=int,
        ),
    ],
    function=convert_pdf_page_to_text,
    return_type=str,
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


def _record_consumed_tokens(
    factory_context: FunctionFactoryContext,
    consumed_tokens: int,
) -> None:
    factory_context.repl_state.usage_ledger.total_consumed_tokens += consumed_tokens


def create_llm_query(factory_context: FunctionFactoryContext) -> Callable[[str], str]:
    def llm_query(text: str) -> str:
        response_text, consumed_tokens = text_query_with_usage(
            factory_context.request_context,
            text,
        )
        _record_consumed_tokens(factory_context, consumed_tokens)
        return response_text

    return llm_query


def create_rlm_query(factory_context: FunctionFactoryContext) -> Callable[[str], str]:
    def rlm_query(prompt: str) -> str:
        try:
            result = recursive_query.execute_recursive_query(
                request=recursive_query.RecursiveQueryRequest(prompt=prompt),
                request_context=factory_context.request_context,
                parent_repl_state=factory_context.repl_state,
                function_collection=factory_context.function_collection,
                config=recursive_query.default_recursive_query_config(),
                runtime=factory_context.recursive_query_runtime,
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error

        _record_consumed_tokens(factory_context, result.total_tokens)
        if result.final_answer is None:
            raise RuntimeError(
                "rlm_query failed: "
                f"termination_reason={result.termination_reason} "
                f"total_iterations={result.total_iterations} "
                f"total_tokens={result.total_tokens}"
            )
        return result.final_answer

    return rlm_query


llm_query_factory = FunctionFactory(
    name="llm_query",
    description="Query the LLM with text input",
    arguments=[Argument(name="text", description="Text input for the LLM", type=str)],
    return_type=str,
    factory=create_llm_query,
)


rlm_query_factory = FunctionFactory(
    name="rlm_query",
    description="""Run a recursive child REPL session for a subproblem.
Usage:
```
question = f"What is the magic number in this chunk?\\n\\n{chunk}"
answer = rlm_query(question)
print(answer)
```
Use this when the subproblem needs its own REPL iterations and tools.
""",
    arguments=[
        Argument(
            name="prompt",
            description="Prompt for the child REPL session",
            type=str,
        )
    ],
    return_type=str,
    factory=create_rlm_query,
)


def create_llm_image_query(
    factory_context: FunctionFactoryContext,
) -> Callable[[str, ImageData], str]:
    def llm_image_query(text: str, image_data: ImageData) -> str:
        response_text, consumed_tokens = image_query_with_usage(
            factory_context.request_context,
            text,
            image_data,
        )
        _record_consumed_tokens(factory_context, consumed_tokens)
        return response_text

    return llm_image_query


llm_image_query_factory = FunctionFactory(
    name="llm_image_query",
    description="""Query the LLM with text and image input.
Usage:
```
result = llm_image_query(
    text="Describe the image",
    image_data=open_image_data("path/to/image.jpg"),
)
print(result)
```
""",
    arguments=[
        Argument(name="text", description="Text input for the LLM", type=str),
        Argument(
            name="image_data",
            description="Image data input for the LLM",
            type=ImageData,
        ),
    ],
    factory=create_llm_image_query,
    return_type=str,
)


llm_image_query_factory = FunctionFactory(
    name="llm_image_query",
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
    factory=create_llm_image_query,
)


def create_llm_pdf_query(
    factory_context: FunctionFactoryContext,
) -> Callable[[str, str, int], str]:
    def llm_pdf_query(text: str, pdf_path: str, page_index: int) -> str:
        image_data = convert_pdf_page_to_image_data(pdf_path, page_index)
        response_text, consumed_tokens = image_query_with_usage(
            factory_context.request_context,
            text,
            image_data,
        )
        _record_consumed_tokens(factory_context, consumed_tokens)
        return response_text

    return llm_pdf_query


llm_pdf_query_factory = FunctionFactory(
    name="llm_pdf_query",
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
    factory=create_llm_pdf_query,
    return_type=str,
)
