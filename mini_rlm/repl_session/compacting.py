from collections.abc import Sequence
from pathlib import Path

from mini_rlm.llm import (
    HistoryItem,
    MessageContent,
    RequestContext,
    TokenUsage,
    get_detailed_token_usage_from_response,
    make_api_request,
)


def compact_history(
    request_context: RequestContext,
    messages: Sequence[HistoryItem],
    *,
    include_context_messages: bool = True,
) -> tuple[list[HistoryItem], TokenUsage]:
    """Compact the message history by sending it to the LLM with a compaction prompt and returning the compacted messages."""
    if request_context.api_type == "responses":
        result = make_api_request(
            request_context,
            messages,
            operation="compact",
            include_context_messages=include_context_messages,
        )
        return list(result.output_items), get_detailed_token_usage_from_response(result)
    prompt_path = Path(__file__).parent.parent / "prompts" / "compaction_prompt.txt"
    with prompt_path.open("r", encoding="utf-8") as f:
        prompt = f.read()
    messages2 = list(messages) + [MessageContent(role="user", content=prompt)]
    result = make_api_request(
        request_context, messages2, include_context_messages=include_context_messages
    )
    token_usage = get_detailed_token_usage_from_response(result)
    if result.messages and len(result.messages) > 0:
        return list(result.messages), token_usage
    else:
        return list(messages)[
            len(messages) // 2 :
        ], token_usage  # fallback: just drop the first half of the messages
