from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommand,
    ReplSessionCommandType,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
    TerminationReason,
)


def _fail_and_exit(
    state: ReplSessionState,
    reason: TerminationReason,
) -> tuple[ReplSessionState, ReplSessionCommand]:
    next_state = state.model_copy(
        update={
            "status": ReplSessionStatus.FAILED,
            "termination_reason": reason,
            "last_command_type": ReplSessionCommandType.EXIT,
        }
    )
    return next_state, ReplSessionCommand(type=ReplSessionCommandType.EXIT)


def _complete_and_exit(
    state: ReplSessionState,
) -> tuple[ReplSessionState, ReplSessionCommand]:
    next_state = state.model_copy(
        update={
            "status": ReplSessionStatus.COMPLETED,
            "is_complete": True,
            "termination_reason": TerminationReason.COMPLETED,
            "last_command_type": ReplSessionCommandType.COMPLETE,
        }
    )
    return next_state, ReplSessionCommand(type=ReplSessionCommandType.COMPLETE)


def _check_termination(
    state: ReplSessionState,
) -> tuple[ReplSessionState, ReplSessionCommand] | None:
    if state.is_cancelled:
        return _fail_and_exit(state, TerminationReason.CANCELLED)

    if state.total_tokens > state.limits.token_limit:
        return _fail_and_exit(state, TerminationReason.TOKEN_LIMIT_EXCEEDED)

    if state.iteration_count >= state.limits.iteration_limit:
        return _fail_and_exit(state, TerminationReason.ITERATIONS_EXHAUSTED)

    elapsed_seconds = state.current_time_seconds - state.started_at_seconds
    if elapsed_seconds > state.limits.timeout_seconds:
        return _fail_and_exit(state, TerminationReason.TIMEOUT)

    if state.error_count >= state.limits.error_threshold:
        return _fail_and_exit(state, TerminationReason.ERROR_THRESHOLD_EXCEEDED)

    return None


def _with_command(
    state: ReplSessionState,
    command_type: ReplSessionCommandType,
) -> tuple[ReplSessionState, ReplSessionCommand]:
    next_state = state.model_copy(update={"last_command_type": command_type})
    return next_state, ReplSessionCommand(type=command_type)


def _apply_result(state: ReplSessionState, result: CommandResult) -> ReplSessionState:
    next_state = state.model_copy(
        update={
            "total_tokens": state.total_tokens + result.consumed_tokens,
            "is_complete": result.is_complete
            if result.is_complete is not None
            else state.is_complete,
        }
    )
    return next_state


def _next_command_after_success(
    state: ReplSessionState,
    prev_command_result: CommandResult,
) -> tuple[ReplSessionState, ReplSessionCommand]:
    command_type = prev_command_result.command_type
    if command_type == ReplSessionCommandType.CALL_LLM:
        new_state, next_command = _with_command(
            state, ReplSessionCommandType.EXECUTE_CODE
        )
        new_state = new_state.model_copy(
            update={
                "last_llm_message": prev_command_result.last_llm_message,
                "repl_results": None,
            }
        )
        return new_state, next_command

    if command_type == ReplSessionCommandType.EXECUTE_CODE:
        new_state, next_command = _with_command(
            state, ReplSessionCommandType.APPEND_HISTORY
        )
        new_state = new_state.model_copy(
            update={"repl_results": prev_command_result.repl_results}
        )
        return new_state, next_command

    if command_type == ReplSessionCommandType.APPEND_HISTORY:
        new_state, next_command = _with_command(
            state, ReplSessionCommandType.CHECK_COMPLETE
        )
        old_messages = state.messages if state.messages is not None else []
        new_state = new_state.model_copy(
            update={"messages": old_messages + (prev_command_result.new_messages or [])}
        )
        return new_state, next_command

    if command_type == ReplSessionCommandType.CHECK_COMPLETE:
        if prev_command_result.is_complete:
            new_state = state.model_copy(
                update={"final_answer": prev_command_result.final_answer}
            )
            return _complete_and_exit(new_state)

        next_state = state.model_copy(
            update={"iteration_count": state.iteration_count + 1}
        )
        if len(next_state.messages or []) > next_state.limits.history_limit:
            return _with_command(next_state, ReplSessionCommandType.COMPACTING)
        return _with_command(next_state, ReplSessionCommandType.CALL_LLM)

    if command_type == ReplSessionCommandType.COMPACTING:
        new_state, next_command = _with_command(state, ReplSessionCommandType.CALL_LLM)
        assert prev_command_result.compacted_messages is not None, (
            "Compacting command result must include compacted_messages"
        )
        new_state = new_state.model_copy(
            update={"messages": prev_command_result.compacted_messages}
        )
        return new_state, next_command

    return _with_command(state, ReplSessionCommandType.EXIT)


def reduce_repl_session(
    prev_state: ReplSessionState,
    prev_command_result: CommandResult | None,
) -> tuple[ReplSessionState, ReplSessionCommand]:
    state = prev_state

    if prev_command_result is not None:
        state = _apply_result(state, prev_command_result)

        if prev_command_result.type != ReplSessionResultType.SUCCESS:
            state = state.model_copy(update={"error_count": state.error_count + 1})
            check = _check_termination(state)
            if check is not None:
                return check
            return _with_command(state, prev_command_result.command_type)

    check = _check_termination(state)
    if check is not None:
        return check

    if prev_command_result is None:
        if len(state.messages or []) > state.limits.history_limit:
            return _with_command(state, ReplSessionCommandType.COMPACTING)
        return _with_command(state, ReplSessionCommandType.CALL_LLM)

    return _next_command_after_success(state, prev_command_result)
