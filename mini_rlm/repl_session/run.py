from datetime import datetime

from mini_rlm.llm import RequestContext
from mini_rlm.repl_session.data_model import (
    ReplSessionLimits,
    ReplSessionResult,
    TerminationReason,
)
from mini_rlm.repl_session.executor import execute_repl_session_loop
from mini_rlm.repl_setup import ReplContext


def run_repl_session(
    repl_context: ReplContext,
    prompt: str,
    limits: ReplSessionLimits | None = None,
    request_context: RequestContext | None = None,
) -> ReplSessionResult:
    """Runs a REPL session with the given prompt and request context, returning the final result."""
    last_state = execute_repl_session_loop(
        repl_context=repl_context,
        prompt=prompt,
        limits=limits,
        request_context=request_context,
    )
    if last_state.ended_at_seconds is None:
        ended_at_seconds = datetime.now().timestamp()
    else:
        ended_at_seconds = last_state.ended_at_seconds
    total_time_seconds = ended_at_seconds - last_state.started_at_seconds

    return ReplSessionResult(
        termination_reason=last_state.termination_reason or TerminationReason.UNKNOWN,
        final_answer=last_state.final_answer,
        total_iterations=last_state.iteration_count,
        total_tokens=last_state.total_tokens,
        total_time_seconds=total_time_seconds,
        repl_history=last_state.repl_history,
    )
