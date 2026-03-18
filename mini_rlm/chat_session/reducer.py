from mini_rlm.chat_session.data_model import (
    ChatDecisionType,
    ChatSessionCommand,
    ChatSessionCommandType,
    ChatSessionResultType,
    ChatSessionState,
    ChatSessionStatus,
    ChatTurn,
    CommandResult,
)
from mini_rlm.llm import merge_model_token_usages


def _with_command(
    state: ChatSessionState,
    command_type: ChatSessionCommandType,
) -> tuple[ChatSessionState, ChatSessionCommand]:
    next_state = state.model_copy(update={"status": ChatSessionStatus.RUNNING})
    return next_state, ChatSessionCommand(type=command_type)


def _apply_result(state: ChatSessionState, result: CommandResult) -> ChatSessionState:
    return state.model_copy(
        update={
            "total_tokens": state.total_tokens + result.consumed_tokens,
            "model_token_usages": merge_model_token_usages(
                state.model_token_usages,
                result.model_token_usages,
            ),
            "last_error": result.error_message,
        }
    )


def _append_turn(
    state: ChatSessionState,
    assistant_text: str,
    result: CommandResult | None = None,
) -> ChatSessionState:
    pending_user_text = state.pending_user_text or ""
    pending_decision = state.pending_decision
    selected_files = pending_decision.file_names if pending_decision is not None else []
    decision_type = (
        pending_decision.type
        if pending_decision is not None
        else result.decision.type
        if result is not None and result.decision is not None
        else ChatDecisionType.RESPOND_CHAT
    )
    reason = pending_decision.reason if pending_decision is not None else None
    next_turn = ChatTurn(
        user_text=pending_user_text,
        assistant_text=assistant_text,
        decision_type=decision_type,
        selected_files=selected_files,
        reason=reason,
        run_summary=result.run_summary if result is not None else None,
    )
    return state.model_copy(
        update={
            "status": ChatSessionStatus.IDLE,
            "pending_user_text": None,
            "pending_decision": None,
            "turns": state.turns + [next_turn],
        }
    )


def reduce_chat_session(
    prev_state: ChatSessionState,
    prev_command_result: CommandResult | None,
) -> tuple[ChatSessionState, ChatSessionCommand]:
    # One chat turn is modeled as a short state machine:
    #   start -> DECIDE -> RUN_AGENT? -> COMPLETE_TURN
    # `COMPLETE_TURN` means "return control to the outer chat loop", not
    # "terminate the whole chat session".
    state = prev_state

    if prev_command_result is None:
        # Entering a new turn.
        # - No pending user text: nothing to do.
        # - Forced run already injected a decision: skip straight to RUN_AGENT.
        # - Otherwise ask the chat model whether to respond directly or run.
        if state.pending_user_text is None or state.pending_user_text.strip() == "":
            return _with_command(
                state.model_copy(update={"status": ChatSessionStatus.IDLE}),
                ChatSessionCommandType.COMPLETE_TURN,
            )
        if state.pending_decision is not None:
            return _with_command(state, ChatSessionCommandType.RUN_AGENT)
        return _with_command(state, ChatSessionCommandType.DECIDE)

    state = _apply_result(state, prev_command_result)
    if prev_command_result.type != ChatSessionResultType.SUCCESS:
        # Any command failure is converted into an assistant-visible turn and
        # the current turn is considered complete.
        assistant_text = (
            prev_command_result.error_message
            or "The chat session failed to produce a valid response."
        )
        return (
            _append_turn(state, assistant_text, prev_command_result),
            ChatSessionCommand(type=ChatSessionCommandType.COMPLETE_TURN),
        )

    if prev_command_result.command_type == ChatSessionCommandType.DECIDE:
        # DECIDE resolved the orchestration question for this turn.
        # - respond_chat: append the assistant message and finish the turn.
        # - run_agent: persist the decision and advance to RUN_AGENT.
        if prev_command_result.decision is None:
            return (
                _append_turn(
                    state,
                    "The chat session failed to produce a valid decision.",
                    prev_command_result,
                ),
                ChatSessionCommand(type=ChatSessionCommandType.COMPLETE_TURN),
            )
        state = state.model_copy(
            update={"pending_decision": prev_command_result.decision}
        )
        if prev_command_result.decision.type == ChatDecisionType.RESPOND_CHAT:
            return (
                _append_turn(
                    state,
                    prev_command_result.decision.message or "",
                    prev_command_result,
                ),
                ChatSessionCommand(type=ChatSessionCommandType.COMPLETE_TURN),
            )
        return _with_command(state, ChatSessionCommandType.RUN_AGENT)

    if prev_command_result.command_type == ChatSessionCommandType.RUN_AGENT:
        # RUN_AGENT already produced the final assistant-facing answer for this
        # turn, so append it and return control to the outer chat loop.
        assistant_text = prev_command_result.assistant_text or ""
        return (
            _append_turn(state, assistant_text, prev_command_result),
            ChatSessionCommand(type=ChatSessionCommandType.COMPLETE_TURN),
        )

    # Fallback: if an unexpected command result appears, stop this turn
    # cleanly rather than leaving the session in RUNNING.
    return (
        state.model_copy(update={"status": ChatSessionStatus.IDLE}),
        ChatSessionCommand(type=ChatSessionCommandType.COMPLETE_TURN),
    )
