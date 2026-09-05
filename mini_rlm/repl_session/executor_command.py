from typing import List

from mini_rlm.code_block import (
    find_code_blocks,
    find_final_answer,
    format_execution_result,
)
from mini_rlm.custom_functions import FunctionCollection
from mini_rlm.debug_logger import get_logger
from mini_rlm.llm import (
    HistoryItem,
    MessageContent,
    ModelTokenUsage,
    RequestContext,
    convert_messages_str,
    get_detailed_token_usage_from_response,
    make_api_request,
    merge_model_token_usages,
)
from mini_rlm.repl import ReplState, execute_code
from mini_rlm.repl_session.compacting import compact_history
from mini_rlm.repl_session.convert import build_session_messages
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommand,
    ReplSessionHistoryEntry,
    ReplSessionResultType,
    ReplSessionState,
)
from mini_rlm.system_prompt import create_system_prompt


def execute_call_llm(
    command: ReplSessionCommand,
    request_context: RequestContext,
    session_state: ReplSessionState,
    function_collection: FunctionCollection | None = None,
) -> CommandResult:
    """Execute a CALL_LLM command by making an API request to the LLM with the current session history and returning the result."""
    messages = build_session_messages(
        session_state, request_context, create_system_prompt(function_collection)
    )
    try:
        res = make_api_request(
            context=request_context, messages=messages, include_context_messages=False
        )
    except (RuntimeError, ValueError) as error:
        return CommandResult(
            type=ReplSessionResultType.ERROR,
            command_type=command.type,
            error_message=str(error),
        )
    if len(res.messages) == 0:
        return CommandResult(
            # error result
            type=ReplSessionResultType.ERROR,
            command_type=command.type,
            error_message="No messages returned from LLM",
        )
    last_message = convert_messages_str(res.messages)
    token_usage = get_detailed_token_usage_from_response(res)
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        consumed_tokens=token_usage.total_tokens,
        model_token_usages=token_usage.model_token_usages,
        last_llm_message=last_message,
        last_llm_items=list(res.output_items)
        if request_context.api_type == "responses"
        else list(res.messages),
    )


def execute_execute_command(
    command: ReplSessionCommand,
    repl: ReplState,
    session_state: ReplSessionState,
) -> CommandResult:
    """Execute an EXECUTE_CODE command by executing the code in the last LLM message and returning the result."""
    if session_state.last_llm_message is None:
        return CommandResult(
            type=ReplSessionResultType.ERROR,
            command_type=command.type,
            error_message="No LLM message to execute",
        )
    logger = get_logger()
    logger.debug(
        "Executing code from last LLM message:\n%s", session_state.last_llm_message
    )
    code_blocks = find_code_blocks(session_state.last_llm_message)
    results = []
    consumed_tokens = 0
    model_token_usages: list[ModelTokenUsage] = []
    for code in code_blocks:
        exec_result = execute_code(state=repl, code=code)
        consumed_tokens += exec_result.consumed_tokens
        model_token_usages = merge_model_token_usages(
            model_token_usages,
            exec_result.model_token_usages,
        )
        results.append(ReplSessionHistoryEntry(code=code, repl_result=exec_result))
    if not code_blocks:
        logger.warning(
            "No code blocks found in last LLM message to execute. Returning empty results. \n%s",
            session_state.last_llm_message,
        )
    else:
        logger.debug("Executed %d code blocks from last LLM message", len(code_blocks))
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        consumed_tokens=consumed_tokens,
        model_token_usages=model_token_usages,
        repl_results=results,
    )


def execute_append_history(
    command: ReplSessionCommand,
    session_state: ReplSessionState,
) -> CommandResult:
    """Execute an APPEND_HISTORY command by appending the last LLM message and execution results to the session history."""
    message = session_state.last_llm_message
    iteration = session_state.repl_results or []
    if message is None:
        return CommandResult(
            type=ReplSessionResultType.ERROR,
            command_type=command.type,
            error_message="No LLM message to append to history",
        )
    new_messages = format_iteration(
        message, iteration, response_items=session_state.last_llm_items
    )
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        new_messages=new_messages,
    )


def format_iteration(
    message: str,
    iteration: List[ReplSessionHistoryEntry],
    max_character_length: int = 20000,
    *,
    response_items: list[HistoryItem] | None = None,
) -> list[HistoryItem]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.
    """
    messages: list[HistoryItem] = (
        list(response_items)
        if response_items
        else [MessageContent(role="assistant", content=message)]
    )

    for code_block in iteration:
        code = code_block.code
        result = code_block.repl_result
        if result is None:
            continue
        result_str = format_execution_result(result)
        if len(result_str) > max_character_length:
            result_str = (
                result_str[:max_character_length]
                + f"... + [{len(result_str) - max_character_length} chars...]"
            )
        execution_message = MessageContent(
            role="user",
            content=f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}",
        )
        messages.append(execution_message)
    return messages


def execute_check_complete(
    command: ReplSessionCommand,
    session_state: ReplSessionState,
    repl_state: ReplState,
) -> CommandResult:
    """Execute a CHECK_COMPLETE command by checking if the last LLM message indicates completion and returning the result."""
    final_answer = None
    for entry in session_state.repl_results or []:
        if entry.repl_result is not None and entry.repl_result.final_answer is not None:
            final_answer = entry.repl_result.final_answer
            break
    if final_answer is None:
        # If no final answer found in execution results, also check the last LLM message for a FINAL(...) statement
        if session_state.last_llm_message is not None:
            final_answer = find_final_answer(
                session_state.last_llm_message, repl_state=repl_state
            )
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        is_complete=final_answer is not None,
        final_answer=final_answer,
    )


def execute_compacting(
    command: ReplSessionCommand,
    session_state: ReplSessionState,
    request_context: RequestContext,
    function_collection: FunctionCollection | None = None,
) -> CommandResult:
    """Execute a COMPACTING command by compacting the session history and returning the result."""
    logger = get_logger()
    messages = build_session_messages(
        session_state, request_context, create_system_prompt(function_collection)
    )
    if session_state.is_compaction_limit_exceeded():
        logger.debug(
            "Total tokens %d exceeded compacting threshold. Compacting history...",
            session_state.total_tokens,
        )
        try:
            new_messages, token_usage = compact_history(
                request_context, messages, include_context_messages=False
            )
        except (RuntimeError, ValueError) as error:
            return CommandResult(
                type=ReplSessionResultType.ERROR,
                command_type=command.type,
                error_message=str(error),
            )
        logger.debug(
            "Compacted history from %d messages to %d messages",
            len(messages),
            len(new_messages),
        )
        return CommandResult(
            type=ReplSessionResultType.SUCCESS,
            command_type=command.type,
            compacted_messages=new_messages,
            history_includes_prompt=request_context.api_type == "responses",
            consumed_tokens=token_usage.total_tokens,
            model_token_usages=token_usage.model_token_usages,
        )
    else:
        return CommandResult(
            type=ReplSessionResultType.SUCCESS,
            command_type=command.type,
            compacted_messages=list(session_state.messages or []),
            history_includes_prompt=session_state.history_includes_prompt,
        )
