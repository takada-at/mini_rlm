import re
from typing import List

from mini_rlm.image import ImageData, convert_image_data_to_image_url
from mini_rlm.llm.api_request import make_api_request
from mini_rlm.llm.data_model import (
    APIRequestResult,
    ImageURL,
    MessageContent,
    MessageContentPart,
    RequestContext,
    TokenUsage,
)
from mini_rlm.llm.token_usage import get_detailed_token_usage_from_response


def message_content_parts_to_text(content_part: MessageContentPart) -> str:
    """Convert a MessageContentPart to text for display."""
    if content_part.type == "text" and content_part.text:
        return content_part.text
    elif content_part.type == "image_url" and content_part.image_url:
        return f"[Image: {content_part.image_url.url}]"
    else:
        raise ValueError(f"Unsupported content part type: {content_part.type}")


def remove_think_tag_contents(text: str) -> str:
    """Remove <think>*</think> contents from the text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def message_content_to_text(content: str | List[MessageContentPart]) -> str:
    """Convert a MessageContent to text for display."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(message_content_parts_to_text(part) for part in content)
    else:
        raise ValueError("Unsupported content type")


def _response_to_text_and_usage(response: APIRequestResult) -> tuple[str, TokenUsage]:
    response_messages = response.messages
    token_usage = get_detailed_token_usage_from_response(response)
    if response_messages:
        return (
            remove_think_tag_contents(
                message_content_to_text(response_messages[0].content)
            ),
            token_usage,
        )
    else:
        return "", token_usage


def text_query_with_usage(context: RequestContext, text: str) -> tuple[str, TokenUsage]:
    """Make a text query to the LLM API and return the response text with token usage."""
    messages = [MessageContent(role="user", content=text)]
    response = make_api_request(context, messages)
    return _response_to_text_and_usage(response)


def text_query(context: RequestContext, text: str) -> str:
    """Make a text query to the LLM API."""
    response_text, _ = text_query_with_usage(context, text)
    return response_text


def image_query_with_usage(
    context: RequestContext,
    text: str,
    image_data: ImageData,
) -> tuple[str, TokenUsage]:
    """Make an image query to the LLM API and return the response text with token usage."""
    image_url = convert_image_data_to_image_url(image_data)
    message_content = [
        MessageContentPart(type="text", text=text),
        MessageContentPart(type="image_url", image_url=ImageURL(url=image_url)),
    ]
    messages = [MessageContent(role="user", content=message_content)]
    response = make_api_request(context, messages)
    return _response_to_text_and_usage(response)


def image_query(context: RequestContext, text: str, image_data: ImageData) -> str:
    """Make an image query to the LLM API."""
    response_text, _ = image_query_with_usage(context, text, image_data)
    return response_text
