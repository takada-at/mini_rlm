from typing import List

from mini_rlm.code_block.parser import find_code_blocks, format_execution_result
from mini_rlm.llm.api_request import make_api_request
from mini_rlm.llm.convert import convert_messages_str
from mini_rlm.llm.data_model import MessageContent, RequestContext
from mini_rlm.repl.data_model import ReplState
from mini_rlm.repl.repl import execute_code
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommand,
    ReplSessionHistoryEntry,
    ReplSessionResultType,
    ReplSessionState,
)


def execute_call_llm(
    command: ReplSessionCommand,
    request_context: RequestContext,
    session_state: ReplSessionState,
) -> CommandResult:
    """Execute a CALL_LLM command by making an API request to the LLM with the current session history and returning the result."""
    system_message = MessageContent(
        role="system",
        content="You are a helpful assistant for executing code and managing a REPL session.",
    )
    history = session_state.messages or []
    messages = [system_message] + history
    res = make_api_request(
        context=request_context,
        messages=messages,
    )
    if len(res.messages) == 0:
        return CommandResult(
            # error result
            type=ReplSessionResultType.ERROR,
            command_type=command.type,
            error_message="No messages returned from LLM",
        )
    last_message = convert_messages_str(res.messages)
    total_tokens = 0
    if "usage" in res.response_json:
        if "total_tokens" in res.response_json["usage"]:
            total_tokens = res.response_json["usage"]["total_tokens"]
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        consumed_tokens=total_tokens,
        last_llm_message=last_message,
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
    code_blocks = find_code_blocks(session_state.last_llm_message)
    results = []
    for code in code_blocks:
        exec_result = execute_code(state=repl, code=code)
        results.append(ReplSessionHistoryEntry(code=code, repl_result=exec_result))
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
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
    new_messages = format_iteration(message, iteration)
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        new_messages=new_messages,
    )


def format_iteration(
    message: str,
    iteration: List[ReplSessionHistoryEntry],
    max_character_length: int = 20000,
) -> list[MessageContent]:
    """
    Format an RLM iteration (including all code blocks) to append to the message history for
    the prompt of the LM in the next iteration. We also truncate code execution results
    that exceed the max_character_length.
    """
    messages = [MessageContent(role="assistant", content=message)]

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
) -> CommandResult:
    """Execute a CHECK_COMPLETE command by checking if the last LLM message indicates completion and returning the result."""
    final_answer = None
    for entry in session_state.repl_results or []:
        if entry.repl_result is not None and entry.repl_result.final_answer is not None:
            final_answer = entry.repl_result.final_answer
    return CommandResult(
        type=ReplSessionResultType.SUCCESS,
        command_type=command.type,
        is_complete=final_answer is not None,
        final_answer=final_answer,
    )


def execute_compacting(
    command: ReplSessionCommand,
    session_state: ReplSessionState,
) -> CommandResult:
    """Execute a COMPACTING command by compacting the session history and returning the result."""
    messages = session_state.messages or []
    history_limit = session_state.limits.history_limit
    if len(messages) >= history_limit:
        # simple compacting strategy: keep only the last N messages, where N is half of the history limit
        new_messages = messages[-(history_limit // 2) :]
        return CommandResult(
            type=ReplSessionResultType.SUCCESS,
            command_type=command.type,
            compacted_messages=new_messages,
        )
    else:
        return CommandResult(
            type=ReplSessionResultType.SUCCESS,
            command_type=command.type,
        )
