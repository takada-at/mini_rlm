from mini_rlm.custom_functions import FunctionCollection
from mini_rlm.llm import RequestContext
from mini_rlm.recursive_query.convert import (
    build_child_recursive_query_runtime,
    build_child_repl_limits,
    extract_inherited_context_payload,
    list_inherited_file_paths,
    resolve_recursive_query_runtime,
)
from mini_rlm.recursive_query.data_model import (
    RecursiveQueryConfig,
    RecursiveQueryRequest,
    RecursiveQueryResult,
    RecursiveQueryRuntime,
)
from mini_rlm.repl import ReplState
from mini_rlm.repl_session import ReplExecutionRequest, execute_repl_session
from mini_rlm.repl_setup import ReplSetupRequest


def execute_recursive_query(
    request: RecursiveQueryRequest,
    request_context: RequestContext,
    parent_repl_state: ReplState,
    function_collection: FunctionCollection | None,
    config: RecursiveQueryConfig,
    runtime: RecursiveQueryRuntime | None,
) -> RecursiveQueryResult:
    active_runtime = resolve_recursive_query_runtime(runtime, config)
    child_runtime = build_child_recursive_query_runtime(active_runtime)
    file_pathes = list_inherited_file_paths(
        parent_repl_state.temp_dir,
        config.inherit_parent_files,
    )
    result = execute_repl_session(
        ReplExecutionRequest(
            prompt=request.prompt,
            setup=ReplSetupRequest(
                request_context=request_context,
                context_payload=extract_inherited_context_payload(
                    parent_repl_state.locals
                ),
                file_paths=file_pathes,
                functions=function_collection,
                recursive_query_runtime=child_runtime,
            ),
            limits=build_child_repl_limits(config),
        )
    )
    return RecursiveQueryResult(
        termination_reason=result.termination_reason.value,
        final_answer=result.final_answer,
        total_iterations=result.total_iterations,
        total_tokens=result.total_tokens,
        model_token_usages=result.model_token_usages,
        total_time_seconds=result.total_time_seconds,
    )
