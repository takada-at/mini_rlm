from pathlib import Path
from typing import List, Tuple

from mini_rlm.llm import (
    MessageContent,
    RequestContext,
    get_token_usage_from_response,
    make_api_request,
)


def compact_history(
    request_context: RequestContext, messages: List[MessageContent]
) -> Tuple[List[MessageContent], int]:
    """Compact the message history by sending it to the LLM with a compaction prompt and returning the compacted messages."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "compaction_prompt.txt"
    with prompt_path.open("r", encoding="utf-8") as f:
        prompt = f.read()
    messages2 = messages + [MessageContent(role="user", content=prompt)]
    result = make_api_request(request_context, messages2)
    total_tokens = get_token_usage_from_response(result)
    if result.messages and len(result.messages) > 0:
        return result.messages, total_tokens
    else:
        return messages[
            len(messages) // 2 :
        ], total_tokens  # fallback: just drop the first half of the messages
