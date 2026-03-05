from collections.abc import Callable

from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommandType,
    ReplSessionResultType,
    ReplSessionState,
)
from mini_rlm.repl_session.reducer import reduce_repl_session

Handler = Callable[[ReplSessionState], CommandResult]


def execute_repl_session_loop(
    initial_state: ReplSessionState,
    run_call_llm: Handler,
    run_execute_code: Handler,
    run_append_history: Handler,
    run_check_complete: Handler,
    run_compacting: Handler,
    now_fn: Callable[[], float],
) -> ReplSessionState:
    state = initial_state
    prev_result: CommandResult | None = None

    while True:
        state = state.model_copy(update={"current_time_seconds": now_fn()})
        state, command = reduce_repl_session(state, prev_result)

        if command.type in (
            ReplSessionCommandType.EXIT,
            ReplSessionCommandType.COMPLETE,
        ):
            return state

        if command.type == ReplSessionCommandType.CALL_LLM:
            prev_result = run_call_llm(state)
            continue

        if command.type == ReplSessionCommandType.EXECUTE_CODE:
            prev_result = run_execute_code(state)
            continue

        if command.type == ReplSessionCommandType.APPEND_HISTORY:
            prev_result = run_append_history(state)
            continue

        if command.type == ReplSessionCommandType.CHECK_COMPLETE:
            prev_result = run_check_complete(state)
            continue

        if command.type == ReplSessionCommandType.COMPACTING:
            prev_result = run_compacting(state)
            continue

        prev_result = CommandResult(
            command_type=command.type,
            type=ReplSessionResultType.ERROR,
            error_message="unknown command",
        )
