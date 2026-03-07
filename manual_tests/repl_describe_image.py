import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from manual_tests.describe_image import (  # noqa: E402
    API_KEY_ENV,
    DEFAULT_PROMPT,
    ENDPOINT_ENV,
    MODEL_ENV,
    create_request_context,
    require_env,
)
from mini_rlm.repl import cleanup  # noqa: E402
from mini_rlm.repl_session import run_repl_session  # noqa: E402
from mini_rlm.repl_setup import setup_repl  # noqa: E402

DEFAULT_IMAGE_PATH = ROOT_DIR / "manual_tests" / "images" / "hello_world.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repl_session to describe an image already added to the REPL."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to the image file to add to the REPL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV),
        help=f"Model name. Defaults to ${MODEL_ENV} when set.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt passed to run_repl_session.",
    )
    return parser.parse_args()


def create_context_payload(image_filename: str) -> dict[str, str]:
    return {
        "image_path": image_filename,
        "note": "The image file in image_path has already been added to the REPL working directory.",
    }


def create_setup_code(image_filename: str) -> str:
    return f"image_path = {image_filename!r}"


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
    image_path = Path(args.image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    endpoint_url = require_env(ENDPOINT_ENV)
    api_key = require_env(API_KEY_ENV)
    request_context = create_request_context(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model=args.model,
    )

    repl_context = setup_repl(
        request_context=request_context,
        file_pathes=[image_path],
    )
    try:
        result = run_repl_session(
            repl_context=repl_context,
            prompt=args.prompt.format(image_path=image_path.name),
        )
    finally:
        cleanup(repl_context.repl_state)

    print_result(
        final_answer=result.final_answer,
        termination_reason=result.termination_reason.value,
        total_iterations=result.total_iterations,
        total_tokens=result.total_tokens,
        total_time_seconds=result.total_time_seconds,
    )


if __name__ == "__main__":
    main()
