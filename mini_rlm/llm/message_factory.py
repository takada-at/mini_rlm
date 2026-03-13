from pathlib import Path

from mini_rlm.image import convert_image_data_to_image_url, open_image_data
from mini_rlm.llm.data_model import ImageURL, MessageContent, MessageContentPart


def create_message_content(
    prompt: str, image_path: Path | None = None
) -> MessageContent:
    """Create a MessageContent object for the given prompt and optional image path."""
    if image_path is None:
        return MessageContent(
            role="user",
            content=[
                MessageContentPart(type="text", text=prompt),
            ],
        )
    image_data = open_image_data(str(image_path))
    image_url = convert_image_data_to_image_url(image_data)
    return MessageContent(
        role="user",
        content=[
            MessageContentPart(type="text", text=prompt),
            MessageContentPart(
                type="image_url",
                image_url=ImageURL(url=image_url, detail="auto"),
            ),
        ],
    )
