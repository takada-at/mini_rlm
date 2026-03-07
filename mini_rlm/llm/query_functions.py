from typing import List

from mini_rlm.image.convert import convert_image_data_to_image_url
from mini_rlm.image.data_model import ImageData
from mini_rlm.llm.api_request import make_api_request
from mini_rlm.llm.data_model import (
    ImageURL,
    MessageContent,
    MessageContentPart,
    RequestContext,
)


def message_content_parts_to_text(content_part: MessageContentPart) -> str:
    """Convert a MessageContentPart to text for display."""
    if content_part.type == "text" and content_part.text:
        return content_part.text
    elif content_part.type == "image_url" and content_part.image_url:
        return f"[Image: {content_part.image_url.url}]"
    else:
        raise ValueError(f"Unsupported content part type: {content_part.type}")


def message_content_to_text(content: str | List[MessageContentPart]) -> str:
    """Convert a MessageContent to text for display."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join(message_content_parts_to_text(part) for part in content)
    else:
        raise ValueError("Unsupported content type")


def text_query(context: RequestContext, text: str) -> str:
    """Make a text query to the LLM API."""
    messages = [MessageContent(role="user", content=text)]
    response = make_api_request(context, messages)
    response_messages = response.messages
    if response_messages:
        return message_content_to_text(response_messages[0].content)
    else:
        return ""


def image_query(context: RequestContext, text: str, image_data: ImageData) -> str:
    """Make an image query to the LLM API."""
    image_url = convert_image_data_to_image_url(image_data)
    message_content = [
        MessageContentPart(type="text", text=text),
        MessageContentPart(type="image_url", image_url=ImageURL(url=image_url)),
    ]
    messages = [MessageContent(role="user", content=message_content)]
    response = make_api_request(context, messages)
    response_messages = response.messages
    if response_messages:
        return message_content_to_text(response_messages[0].content)
    else:
        return ""
