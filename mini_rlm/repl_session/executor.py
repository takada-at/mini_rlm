from collections.abc import Callable
from datetime import datetime

from mini_rlm.debug_logger import get_log_file_path, get_logger
from mini_rlm.llm.data_model import RequestContext
from mini_rlm.repl.data_model import ReplState
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

Handler = Callable[[ReplSessionState], CommandResult]

logger = get_logger()


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
    logger.debug(
        "repl_session.loop state status=%s iteration=%s total_tokens=%s errors=%s prev_result=%s",
        state.status,
        state.iteration_count,
        state.total_tokens,
        state.error_count,
        prev_result.command_type if prev_result else None,
    )


def _log_command(command_type: ReplSessionCommandType) -> None:
    logger.debug("repl_session.command type=%s", command_type)


def _log_result(result: CommandResult) -> None:
    logger.debug(
        "repl_session.result command=%s fields=%s",
        result.command_type,
        _build_result_log_fields(result),
    )


def _log_session_end(state: ReplSessionState) -> None:
    logger.debug(
        "repl_session.end status=%s termination_reason=%s iterations=%s total_tokens=%s final_answer_present=%s",
        state.status,
        state.termination_reason,
        state.iteration_count,
        state.total_tokens,
        state.final_answer is not None,
    )


def execute_repl_session_loop(
    repl: ReplState,
    request_context: RequestContext,
    prompt: str,
) -> ReplSessionState:
    logger.debug(
        "repl_session.start prompt_length=%s log_file=%s",
        len(prompt),
        get_log_file_path(),
    )

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
            _log_session_end(state)
            return state

        if command.type == ReplSessionCommandType.CALL_LLM:
            prev_result = execute_call_llm(command, request_context, state)
            _log_result(prev_result)
            continue

        if command.type == ReplSessionCommandType.EXECUTE_CODE:
            prev_result = execute_execute_command(command, repl, state)
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
