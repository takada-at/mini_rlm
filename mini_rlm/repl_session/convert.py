from mini_rlm.llm import HistoryItem, MessageContent, RequestContext
from mini_rlm.repl_session.data_model import ReplSessionLimits, ReplSessionState


def build_session_messages(
    state: ReplSessionState,
    context: RequestContext,
    system_prompt: str,
) -> list[HistoryItem]:
    """Compose a complete input without duplicating users retained by compaction."""
    prefix = context.messages or []
    if state.history_includes_prompt:
        # /compact retains user messages in its output. Static instructions
        # remain external to that compacted conversation and must be resent.
        prefix = [
            message for message in prefix if message.role in ("system", "developer")
        ]
    messages: list[HistoryItem] = [
        *prefix,
        MessageContent(role="system", content=system_prompt),
    ]
    if not state.history_includes_prompt:
        messages.append(MessageContent(role="user", content=state.prompt))
    messages.extend(state.messages or [])
    return messages


def estimate_session_history_tokens(state: ReplSessionState) -> int:
    """Conservative UTF-8 byte estimate of retained input, not billed usage.

    Includes opaque Responses items and message framing. This model-independent
    estimate is intentionally generous for text and opaque items. Images use a
    configurable allowance independent of URL/base64 length. Neither is an exact
    model tokenizer; configure capacity and image allowance for the model in use.
    """
    prefix = state.input_prefix
    if state.history_includes_prompt:
        prefix = [
            item
            for item in prefix
            if isinstance(item, MessageContent) and item.role in ("system", "developer")
        ]
    messages = list(prefix)
    if not state.history_includes_prompt:
        messages.append(MessageContent(role="user", content=state.prompt))
    messages.extend(state.messages or [])
    return sum(
        _estimate_input_value(
            item.model_dump(exclude_none=True), state.limits.image_token_estimate
        )
        + 16
        for item in messages
    )


def _estimate_input_value(value: object, image_tokens: int) -> int:
    if isinstance(value, dict):
        if value.get("type") in ("image_url", "input_image"):
            # URLs/base64 are transport representations, not image token counts.
            return image_tokens
        return (
            sum(
                len(str(key).encode("utf-8"))
                + _estimate_input_value(item, image_tokens)
                + 4
                for key, item in value.items()
            )
            + 2
        )
    if isinstance(value, list):
        return sum(_estimate_input_value(item, image_tokens) + 1 for item in value) + 2
    return len(str(value).encode("utf-8")) + 2


def resolve_session_limits(
    limits: ReplSessionLimits, context: RequestContext
) -> ReplSessionLimits:
    """Reserve at least the configured generation maximum without mutating inputs."""
    params = context.kwargs or {}
    requested_output = (
        params.get("max_output_tokens")
        if context.api_type == "responses"
        else params.get("max_completion_tokens", params.get("max_tokens"))
    )
    if isinstance(requested_output, int) and not isinstance(requested_output, bool):
        return limits.model_copy(
            update={
                "output_token_reserve": max(
                    limits.output_token_reserve, requested_output
                )
            }
        )
    return limits
