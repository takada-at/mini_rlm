import argparse
import os
from pathlib import Path
from typing import Sequence

from mini_rlm import (
    MessageContent,
    convert_messages_str,
    create_message_content,
    create_request_context,
    make_api_request,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


ENDPOINT_ENV = "MINI_RLM_LLM_ENDPOINT"
API_KEY_ENV = "MINI_RLM_LLM_API_KEY"
MODEL_ENV = "MINI_RLM_LLM_MODEL"
DEFAULT_PROMPT = "Please describe this image({image_path})."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send an image to an LLM and print the description."
    )
    parser.add_argument("image_path", help="Path to the image file to describe.")
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV),
        help=f"Model name. Defaults to ${MODEL_ENV} when set.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text prompt sent together with the image.",
    )
    parser.add_argument(
        "--detail",
        choices=["low", "high", "auto"],
        default="auto",
        help="Image detail hint for OpenAI-compatible multimodal endpoints.",
    )
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


def validate_response(messages: Sequence[MessageContent]) -> str:
    if not messages:
        raise RuntimeError("The LLM response did not include any messages.")
    return convert_messages_str(list(messages)).strip()


def main() -> None:
    args = parse_args()
    endpoint_url = require_env(ENDPOINT_ENV)
    api_key = require_env(API_KEY_ENV)
    context = create_request_context(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model=args.model,
    )
    message = create_message_content(
        image_path=Path(args.image_path),
        prompt=args.prompt.format(image_path=args.image_path),
    )
    result = make_api_request(context=context, messages=[message])
    print(validate_response(result.messages))


if __name__ == "__main__":
    main()
