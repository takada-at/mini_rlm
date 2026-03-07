from collections.abc import Callable
from datetime import datetime

from mini_rlm.debug_logger import get_log_file_path, get_logger
from mini_rlm.llm import RequestContext
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommandType,
    ReplSessionLimits,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
)
from mini_rlm.repl_session.executor_command import (
    execute_append_history,
    execute_call_llm,
    execute_check_complete,
    execute_compacting,
    execute_execute_command,
)
from mini_rlm.repl_session.reducer import reduce_repl_session
from mini_rlm.repl_setup import ReplContext

Handler = Callable[[ReplSessionState], CommandResult]


def _debug(message: str, *args: object) -> None:
    try:
        get_logger().debug(message, *args)
    except OSError:
        return


def _build_result_log_fields(result: CommandResult) -> dict[str, object]:
    return {
        "result_type": result.type,
        "error_message": result.error_message,
        "consumed_tokens": result.consumed_tokens,
        "has_last_llm_message": result.last_llm_message is not None,
        "repl_results_count": len(result.repl_results or []),
        "new_messages_count": len(result.new_messages or []),
        "compacted_messages_count": len(result.compacted_messages or []),
        "is_complete": result.is_complete,
    }


def _log_loop_state(state: ReplSessionState, prev_result: CommandResult | None) -> None:
    _debug(
        "repl_session.loop state status=%s iteration=%s total_tokens=%s errors=%s prev_result=%s",
        state.status,
        state.iteration_count,
        state.total_tokens,
        state.error_count,
        prev_result.command_type if prev_result else None,
    )


def _log_command(command_type: ReplSessionCommandType) -> None:
    _debug("repl_session.command type=%s", command_type)


def _log_result(result: CommandResult) -> None:
    _debug(
        "repl_session.result command=%s fields=%s",
        result.command_type,
        _build_result_log_fields(result),
    )


def _log_session_end(state: ReplSessionState) -> None:
    _debug(
        "repl_session.end status=%s termination_reason=%s iterations=%s total_tokens=%s final_answer_present=%s",
        state.status,
        state.termination_reason,
        state.iteration_count,
        state.total_tokens,
        state.final_answer is not None,
    )


def execute_repl_session_loop(
    repl_context: ReplContext,
    prompt: str,
    request_context: RequestContext | None = None,
) -> ReplSessionState:
    """
    Execute a REPL session loop and return the final ReplSessionState.

    A pure state-machine loop that determines the next command via reduce_repl_session
    and dispatches to the appropriate executor. Continues until a termination condition
    is met (max iterations exceeded, timeout, error threshold exceeded, or explicit
    completion command).

    Session limits:
        - token_limit: 1,000,000 tokens
        - iteration_limit: 100 iterations
        - timeout_seconds: 60 seconds
        - error_threshold: 5 errors
        - history_limit: 50 entries

    Command dispatch flow:
        1. reduce_repl_session determines the next command
        2. EXIT / COMPLETE -> exit the loop and return state
        3. CALL_LLM -> call the LLM
        4. EXECUTE_CODE -> execute code via repl_context.repl_state
        5. APPEND_HISTORY -> append to message history
        6. CHECK_COMPLETE -> check whether the session is complete
        7. COMPACTING -> compact the message history

    Args:
        repl_context (ReplContext): Context holding the REPL state (repl_state).
        prompt (str): The user prompt to start the session with.
        request_context (RequestContext | None): Context holding model settings for LLM requests.
            If specified, you can use a different RequestContext from the ReplSession.

    Returns:
        ReplSessionState: The final state at session end.
            Inspect status, termination_reason, and final_answer to retrieve results.
    """
    _debug(
        "repl_session.start prompt_length=%s log_file=%s",
        len(prompt),
        get_log_file_path(),
    )
    if request_context is None:
        request_context = repl_context.request_context

    state = ReplSessionState(
        prompt=prompt,
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=1_000_000,
            iteration_limit=100,
            timeout_seconds=60.0,
            error_threshold=5,
            history_limit=50,
        ),
        started_at_seconds=datetime.now().timestamp(),
        current_time_seconds=datetime.now().timestamp(),
    )
    prev_result: CommandResult | None = None

    while True:
        state = state.model_copy(
            update={"current_time_seconds": datetime.now().timestamp()}
        )
        _log_loop_state(state, prev_result)
        state, command = reduce_repl_session(state, prev_result)
        _log_command(command.type)

        if command.type in (
            ReplSessionCommandType.EXIT,
            ReplSessionCommandType.COMPLETE,
        ):
            end_state = state.model_copy(
                update={"ended_at_seconds": datetime.now().timestamp()}
            )
            _log_session_end(end_state)
            return end_state

        if command.type == ReplSessionCommandType.CALL_LLM:
            prev_result = execute_call_llm(
                command,
                request_context,
                state,
                function_collection=repl_context.functions,
            )
            _log_result(prev_result)
            continue

        if command.type == ReplSessionCommandType.EXECUTE_CODE:
            prev_result = execute_execute_command(
                command, repl_context.repl_state, state
            )
            _log_result(prev_result)
            continue

        if command.type == ReplSessionCommandType.APPEND_HISTORY:
            prev_result = execute_append_history(command, state)
            _log_result(prev_result)
            continue

        if command.type == ReplSessionCommandType.CHECK_COMPLETE:
            prev_result = execute_check_complete(command, state)
            _log_result(prev_result)
            continue

        if command.type == ReplSessionCommandType.COMPACTING:
            prev_result = execute_compacting(command, state)
            _log_result(prev_result)
            continue

        prev_result = CommandResult(
            command_type=command.type,
            type=ReplSessionResultType.ERROR,
            error_message="unknown command",
        )
        _log_result(prev_result)
