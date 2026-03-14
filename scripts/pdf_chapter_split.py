import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pypdfium2 as pdfium

from mini_rlm import (
    ReplExecutionRequest,
    ReplSessionLimits,
    ReplSessionResult,
    ReplSetupRequest,
    RequestContext,
    create_request_context,
    execute_repl_session,
    pdf_function_collection,
)

PROMPT_PAGES = """The pdf file {pdf_path} has already been added to the REPL working directory. 
(1)Please find the page number where the Chapter {chapter_number} starts and return the page number as an integer(0-indexed).
(2)Please find the page number where the Chapter {chapter_number} ends and return the page number as an integer(0-indexed).
Be cautious when using llm_query—it doesn't possess any information beyond what's in the input prompt. Asking it about the PDF content directly would be pointless.
The page numbers in the table of contents may differ from the actual PDF pages, so please verify the text.
Chapter titles can provide important clues. Using llm_query_pdf to analyze page information is also a highly effective approach.

Flow:
1. First, try to find the page numbers from the table of contents. If the table of contents is not available or doesn't include the chapter, return -1 for the start page or end page respectively.
2. Then, try to find the start page by looking for the chapter title in the page content. If not found, return -1.
3. Finally, try to find the end page by looking for the next chapter title in the page content. If not found, return -1.
4. Check the consistency of the start and end page numbers found from different methods. If there are multiple candidates, use the most common page number. If there are conflicting page numbers, use the one found from the page content. If there is still a conflict, use the one with more evidence.
5. Return the final start and end page numbers in the format of below. If the chapter is not found, return -1 for the start page or end page respectively.

If failure occurs, analyze the cause and record the reason in the `failure_reason` variable.

Answer in the following format:
<start_page_number>,<end_page_number>
"""

MODEL = "openai/gpt-5.3-codex"
SUB_MODEL = "qwen/qwen3.5-35b-a3b"
START_PAGE_PATTERN = re.compile(
    r"\bstart(?:s|ing)?(?:[\s_-]*page(?:[\s_-]*number)?)?\b[^\d-]{0,40}(-?\d+)",
    re.IGNORECASE,
)
END_PAGE_PATTERN = re.compile(
    r"\bend(?:s|ing)?(?:[\s_-]*page(?:[\s_-]*number)?)?\b[^\d-]{0,40}(-?\d+)",
    re.IGNORECASE,
)
PAGE_RANGE_PATTERN = re.compile(r"(-?\d+)\s*,\s*(-?\d+)")


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repl_session to describe a PDF already added to the REPL."
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file to add to the REPL.",
    )
    parser.add_argument(
        "chapter",
        type=int,
        help="Chapter number to find the start and end page numbers for.",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        type=Path,
        help="Path to save the output PDF file or directory. Defaults to the input PDF directory.",
    )
    return parser.parse_args()


def create_context_payload(
    pdf_filename: str, payload: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    res = {
        "pdf_path": pdf_filename,
        "note": "The PDF file in pdf_path has already been added to the REPL working directory.",
    }
    if payload:
        res.update(payload)
    return res


def print_result(
    final_answer: str | None,
    termination_reason: str,
    total_iterations: int,
    total_tokens: int,
    total_time_seconds: float,
) -> None:
    print(f"termination_reason: {termination_reason}")
    print(f"total_iterations: {total_iterations}")
    print(f"total_tokens: {total_tokens}")
    print(f"total_time_seconds: {total_time_seconds:.2f}")
    print()
    if final_answer is None:
        raise RuntimeError("REPL session completed without a final_answer.")
    print(final_answer)


def run_repl(
    request_context: RequestContext,
    request_context2: RequestContext,
    pdf_path: Path,
    prompt: str,
    payload: Optional[Dict[str, str]] = None,
) -> ReplSessionResult:
    limits = ReplSessionLimits(
        token_limit=1_000_000,
        iteration_limit=100,
        timeout_seconds=3600.0,
        error_threshold=5,
    )
    return execute_repl_session(
        ReplExecutionRequest(
            prompt=prompt,
            setup=ReplSetupRequest(
                request_context=request_context2,
                file_paths=[pdf_path],
                context_payload=create_context_payload(pdf_path.name, payload),
                functions=pdf_function_collection(),
            ),
            limits=limits,
            session_request_context=request_context,
        )
    )


def extract_page_number(pattern: re.Pattern[str], final_answer: str) -> Optional[int]:
    match = pattern.search(final_answer)
    if match is None:
        return None
    return int(match.group(1))


def parse_page_number(final_answer: str | None) -> Tuple[int, int]:
    if final_answer is None:
        raise RuntimeError("REPL session completed without a final_answer.")
    start_page = extract_page_number(START_PAGE_PATTERN, final_answer)
    end_page = extract_page_number(END_PAGE_PATTERN, final_answer)
    if start_page is not None and end_page is not None:
        return start_page, end_page

    match = PAGE_RANGE_PATTERN.search(final_answer)
    if not match:
        raise ValueError(
            f"Failed to parse page numbers from LLM response: {final_answer}"
        )
    start_page = int(match.group(1))
    end_page = int(match.group(2))
    return start_page, end_page


def resolve_output_path(
    pdf_path: Path,
    chapter_num: int,
    save_path: Path | None,
) -> Path:
    output_filename = f"{pdf_path.stem}_chapter_{chapter_num}.pdf"
    if save_path is None:
        return pdf_path.parent / output_filename
    if save_path.suffix.lower() == ".pdf":
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path / output_filename


def fetch_page_range(
    request_context: RequestContext,
    request_context2: RequestContext,
    pdf_path: Path,
    chapter_num: int,
) -> tuple[int, int]:
    result0 = run_repl(
        request_context=request_context,
        request_context2=request_context2,
        pdf_path=pdf_path,
        prompt=PROMPT_PAGES.format(pdf_path=pdf_path.name, chapter_number=chapter_num),
    )
    for token_usage in result0.model_token_usages:
        print(
            f"Model: {token_usage.model_name}, Prompt tokens: {token_usage.prompt_tokens}, Completion tokens: {token_usage.completion_tokens}"
        )
    try:
        start_page, end_page = parse_page_number(result0.final_answer)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse start page number from LLM response: {result0.final_answer}"
        ) from e
    return start_page, end_page


def main_task(
    request_context: RequestContext,
    request_context2: RequestContext,
    pdf_path: Path,
    chapter_num: int,
    save_path: Path | None = None,
) -> None:
    if chapter_num < 1:
        raise ValueError(f"chapter_num must be positive: {chapter_num}")

    start_page, end_page = fetch_page_range(
        request_context, request_context2, pdf_path, chapter_num
    )
    print(f"Chapter {chapter_num} page range: {start_page} - {end_page}")
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        total_pages = len(pdf)
        if start_page < 0 or end_page < 0:
            raise ValueError(
                f"Page numbers must be 0-based non-negative integers: {start_page}, {end_page}"
            )
        if start_page > end_page:
            raise ValueError(
                f"Chapter {chapter_num} has an invalid page range: {start_page} - {end_page}"
            )
        if end_page >= total_pages:
            raise ValueError(
                f"Chapter {chapter_num} end page {end_page} exceeds total pages {total_pages}"
            )

        output_path = resolve_output_path(pdf_path, chapter_num, save_path)
        with pdfium.PdfDocument.new() as split_pdf:
            split_pdf.import_pages(pdf, pages=list(range(start_page, end_page + 1)))
            split_pdf.save(output_path)
    print(f"Saved chapter {chapter_num} to {output_path}")


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    endpoint_url = require_env("API_ENDPOINT")
    # e.g. openrouter.aiのエンドポイント:
    # https://openrouter.ai/api/v1/chat/completions
    api_key = require_env("API_KEY")
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = pdf_path.parent
    request_context = create_request_context(
        endpoint_url=endpoint_url,
        model=MODEL,
        api_key=api_key,
    )
    request_context2 = create_request_context(
        endpoint_url=endpoint_url,
        model=SUB_MODEL,
        api_key=api_key,
    )
    main_task(request_context, request_context2, pdf_path, args.chapter, save_path)


if __name__ == "__main__":
    main()
