from collections.abc import Callable
from datetime import datetime

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


def execute_repl_session_loop(
    repl: ReplState,
    request_context: RequestContext,
) -> ReplSessionState:
    state = ReplSessionState(
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
        state, command = reduce_repl_session(state, prev_result)

        if command.type in (
            ReplSessionCommandType.EXIT,
            ReplSessionCommandType.COMPLETE,
        ):
            return state

        if command.type == ReplSessionCommandType.CALL_LLM:
            prev_result = execute_call_llm(command, request_context, state)
            continue

        if command.type == ReplSessionCommandType.EXECUTE_CODE:
            prev_result = execute_execute_command(command, repl, state)
            continue

        if command.type == ReplSessionCommandType.APPEND_HISTORY:
            prev_result = execute_append_history(command, state)
            continue

        if command.type == ReplSessionCommandType.CHECK_COMPLETE:
            prev_result = execute_check_complete(command, state)
            continue

        if command.type == ReplSessionCommandType.COMPACTING:
            prev_result = execute_compacting(command, state)
            continue

        prev_result = CommandResult(
            command_type=command.type,
            type=ReplSessionResultType.ERROR,
            error_message="unknown command",
        )
