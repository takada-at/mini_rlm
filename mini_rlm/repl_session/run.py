from datetime import datetime

from mini_rlm.llm import RequestContext
from mini_rlm.repl import cleanup
from mini_rlm.repl_session.data_model import (
    ReplExecutionRequest,
    ReplSessionLimits,
    ReplSessionResult,
    TerminationReason,
)
from mini_rlm.repl_session.executor import execute_repl_session_loop
from mini_rlm.repl_setup import ReplContext, setup_repl


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
        model_token_usages=last_state.model_token_usages,
        total_time_seconds=total_time_seconds,
        repl_history=last_state.repl_history,
    )


def execute_repl_session(request: ReplExecutionRequest) -> ReplSessionResult:
    """Set up a REPL, run a session, and always clean up the REPL state."""
    repl_context = setup_repl(
        request_context=request.setup.request_context,
        setup_code=request.setup.setup_code,
        context_payload=request.setup.context_payload,
        file_pathes=request.setup.file_paths,
        files=request.setup.files,
        functions=request.setup.functions,
        recursive_query_runtime=request.setup.recursive_query_runtime,
    )
    try:
        return run_repl_session(
            repl_context=repl_context,
            prompt=request.prompt,
            limits=request.limits,
            request_context=request.session_request_context,
        )
    finally:
        cleanup(repl_context.repl_state)
