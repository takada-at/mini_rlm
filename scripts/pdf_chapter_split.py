import argparse
import os
import re
from pathlib import Path

import pypdfium2 as pdfium
import requests

from mini_rlm.custom_functions import pdf_function_collection
from mini_rlm.llm import Endpoint, RequestContext
from mini_rlm.repl import cleanup
from mini_rlm.repl_session import ReplSessionLimits, ReplSessionResult, run_repl_session
from mini_rlm.repl_setup import setup_repl

PROMPT_TOC_PAGE = """The pdf file {pdf_path} has already been added to the REPL working directory. 
Please find the page number where the the table of contents starts and return the page number as an integer.
From the table of contents, locate the starting page of Chapter {chapter_number} and the next chapter's starting page, then report your findings.
"""

PROMPT_PAGE_START = """The pdf file {pdf_path} has already been added to the REPL working directory. 
Please find the page number where the Chapter {chapter_number} starts and return the page number as an integer.
The following information was obtained from the table of contents. Note that the page numbers in the table of contents often differ from the actual PDF page numbers, so be sure to verify the information before providing the number.

# Report
{toc_report}
"""

PROMPT_PAGE_END = """The pdf file {pdf_path} has already been added to the REPL working directory. 
Please find the page number where the Chapter {chapter_number} ends and return the page number as an integer.
The following information was obtained from the table of contents. Note that the page numbers in the table of contents often differ from the actual PDF page numbers, so be sure to verify the information before providing the number.

# Report
{toc_report}
"""

MODEL = "openai/gpt-5.3-codex"
SUB_MODEL = "qwen/qwen3.5-35b-a3b"


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


def create_context_payload(pdf_filename: str) -> dict[str, str]:
    return {
        "pdf_path": pdf_filename,
        "note": "The PDF file in pdf_path has already been added to the REPL working directory.",
    }


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
) -> ReplSessionResult:
    repl_context = setup_repl(
        request_context=request_context2,
        file_pathes=[pdf_path],
        context_payload=create_context_payload(pdf_path.name),
        functions=pdf_function_collection(),
    )
    limits = ReplSessionLimits(
        token_limit=1_000_000,
        iteration_limit=100,
        timeout_seconds=3600.0,
        error_threshold=5,
        history_limit=50,
    )
    try:
        return run_repl_session(
            repl_context=repl_context,
            prompt=prompt,
            limits=limits,
            request_context=request_context,
        )
    finally:
        cleanup(repl_context.repl_state)


def parse_page_number(final_answer: str | None) -> int:
    if final_answer is None:
        raise RuntimeError("REPL session completed without a final_answer.")
    match = re.search(r"\d+", final_answer)
    if match is None:
        raise RuntimeError(
            f"Failed to parse page number from LLM response: {final_answer}"
        )
    return int(match.group())


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
        prompt=PROMPT_TOC_PAGE.format(
            pdf_path=pdf_path.name, chapter_number=chapter_num
        ),
    )
    toc_report = result0.final_answer
    print(f"Table of contents report for chapter {chapter_num}: {toc_report}")
    result1 = run_repl(
        request_context=request_context,
        request_context2=request_context2,
        pdf_path=pdf_path,
        prompt=PROMPT_PAGE_START.format(
            pdf_path=pdf_path.name, chapter_number=chapter_num, toc_report=toc_report
        ),
    )
    try:
        start_page = parse_page_number(result1.final_answer)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse start page number from LLM response: {result1.final_answer}"
        ) from e
    print(f"Chapter {chapter_num} starts at page {start_page}")
    result2 = run_repl(
        request_context=request_context,
        request_context2=request_context2,
        pdf_path=pdf_path,
        prompt=PROMPT_PAGE_END.format(
            pdf_path=pdf_path.name, chapter_number=chapter_num, toc_report=toc_report
        ),
    )
    try:
        end_page = parse_page_number(result2.final_answer)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse end page number from LLM response: {result2.final_answer}"
        ) from e
    print(f"Chapter {chapter_num} ends at page {end_page}")
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
    api_key = require_env("API_KEY")
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = pdf_path.parent
    request_context = RequestContext(
        session=requests.Session(),
        endpoint=Endpoint(
            url=endpoint_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ),
        kwargs={
            "model": MODEL,
        },
    )
    request_context2 = RequestContext(
        session=requests.Session(),
        endpoint=Endpoint(
            url=endpoint_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ),
        kwargs={
            "model": SUB_MODEL,
        },
    )
    main_task(request_context, request_context2, pdf_path, args.chapter, save_path)


if __name__ == "__main__":
    main()
