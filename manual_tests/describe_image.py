import argparse
import os
import sys
from pathlib import Path
from typing import Literal, Sequence, cast

import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mini_rlm.image.convert import (  # noqa: E402
    convert_image_data_to_image_url,
    open_image_data,
)
from mini_rlm.llm import (  # noqa: E402
    Endpoint,
    ImageURL,
    MessageContent,
    MessageContentPart,
    RequestContext,
    convert_messages_str,
    make_api_request,
)

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


def create_request_context(
    endpoint_url: str, api_key: str, model: str | None
) -> RequestContext:
    kwargs = {}
    if model is not None:
        kwargs["model"] = model
    return RequestContext(
        session=requests.Session(),
        endpoint=Endpoint(
            url=endpoint_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ),
        kwargs=kwargs,
    )


def create_messages(
    image_path: str,
    prompt: str,
    detail: str,
) -> list[MessageContent]:
    image_data = open_image_data(image_path)
    image_url = convert_image_data_to_image_url(image_data)
    assert detail in ["low", "high", "auto"], "Invalid detail value"
    return [
        MessageContent(
            role="user",
            content=[
                MessageContentPart(type="text", text=prompt),
                MessageContentPart(  # type: ignore
                    type="image_url",
                    image_url=ImageURL(
                        url=image_url,
                        detail=cast(Literal["low", "high", "auto"], detail),
                    ),  # type: ignore
                ),
            ],
        )
    ]


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
    messages = create_messages(
        image_path=args.image_path,
        prompt=args.prompt.format(image_path=args.image_path),
        detail=args.detail,
    )
    result = make_api_request(context=context, messages=messages)
    print(validate_response(result.messages))


if __name__ == "__main__":
    main()
