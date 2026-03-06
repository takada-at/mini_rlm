from typing import List

from mini_rlm.llm.data_model import MessageContent


def convert_messages_str(messages: List[MessageContent]) -> str:
    """Convert a list of MessageContent to a string for logging or debugging."""
    res = ""
    for message in messages:
        if isinstance(message.content, str):
            res += message.content
        else:
            for part in message.content:
                if hasattr(part, "text") and isinstance(part.text, str):
                    res += part.text
    return res
