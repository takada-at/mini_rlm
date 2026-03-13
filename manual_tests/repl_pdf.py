import argparse
import os
from pathlib import Path

from manual_tests.describe_image import (  # noqa: E402
    require_env,
)
from mini_rlm import (
    ReplExecutionRequest,
    ReplSessionLimits,
    ReplSetupRequest,
    create_request_context,
    execute_repl_session,
    pdf_function_collection,
)

DEFAULT_PROMPT = """The pdf file {pdf_path} has already been added to the REPL working directory. 
Please find the page number where the Chapter 2 starts and return the page number as an integer.
In this environment, LLM calls are extremely slow, so please measure the time and call them carefully.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repl_session to describe a PDF already added to the REPL."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the PDF file to add to the REPL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINI_RLM_LLM_MODEL"),
        help="Model name. Defaults to $MINI_RLM_LLM_MODEL when set.",
    )
    parser.add_argument(
        "--sub_model",
        default=os.environ.get("MINI_RLM_LLM_MODEL"),
        help="Sub-model name. Defaults to $MINI_RLM_LLM_MODEL when set.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt passed to run_repl_session.",
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


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    endpoint_url = require_env("MINI_RLM_LLM_ENDPOINT")
    api_key = require_env("MINI_RLM_LLM_API_KEY")
    sub_endpoint_url = os.environ.get("MINI_RLM_LLM_SUB_ENDPOINT", endpoint_url)
    sub_api_key = os.environ.get("MINI_RLM_LLM_SUB_API_KEY", api_key)
    if args.sub_model:
        sub_model = args.sub_model
    elif args.model:
        sub_model = args.model
    else:
        raise RuntimeError(
            "Model name must be specified via --sub_model, --model, or $MINI_RLM_LLM_MODEL environment variable."
        )
    print(f"Using endpoint: {endpoint_url}")
    print(f"Using sub-endpoint: {sub_endpoint_url}")
    print(f"Using model: {args.model}")
    print(f"Using sub-model: {sub_model}")
    request_context_sub = create_request_context(
        endpoint_url=sub_endpoint_url,
        api_key=sub_api_key,
        model=sub_model,
    )
    if sub_endpoint_url != endpoint_url:
        request_context_main = create_request_context(
            endpoint_url=endpoint_url,
            api_key=api_key,
            model=args.model,
        )
    else:
        request_context_main = request_context_sub
    limits = ReplSessionLimits(
        token_limit=1_000_000,
        iteration_limit=100,
        timeout_seconds=3600.0,
        error_threshold=5,
    )
    result = execute_repl_session(
        ReplExecutionRequest(
            prompt=args.prompt.format(pdf_path=pdf_path.name),
            setup=ReplSetupRequest(
                request_context=request_context_sub,
                file_paths=[pdf_path],
                context_payload=create_context_payload(pdf_path.name),
                functions=pdf_function_collection(),
            ),
            limits=limits,
            session_request_context=request_context_main,
        )
    )

    print_result(
        final_answer=result.final_answer,
        termination_reason=result.termination_reason.value,
        total_iterations=result.total_iterations,
        total_tokens=result.total_tokens,
        total_time_seconds=result.total_time_seconds,
    )


if __name__ == "__main__":
    main()
