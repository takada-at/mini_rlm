from mini_rlm.llm import HistoryItem, MessageContent, RequestContext
from mini_rlm.repl_session.data_model import ReplSessionState


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
