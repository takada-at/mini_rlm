import sys

from mini_rlm.chat_session import (
    RunSummary,
    build_run_context_payload,
    create_chat_session,
    execute_chat_turn,
    run_chat_session,
)
from mini_rlm.cli.convert import (
    build_request_context,
    build_run_prompt,
    convert_files_to_attachments,
    format_run_summary,
    select_function_collection,
)
from mini_rlm.cli.data_model import ChatCLIConfig, RunCLIConfig
from mini_rlm.repl_session import ReplExecutionRequest, execute_repl_session
from mini_rlm.repl_setup import ReplFileRef, ReplSetupRequest


def _read_non_interactive_prompt() -> str | None:
    if sys.stdin.isatty():
        return None
    prompt = sys.stdin.read().strip()
    if prompt == "":
        return None
    return prompt


def _print_run_summary(summary: RunSummary) -> None:
    print(format_run_summary(summary))
    print()


def run_chat_command(config: ChatCLIConfig) -> int:
    attachments = convert_files_to_attachments(config.files)
    chat_request_context = build_request_context(
        endpoint_url=config.endpoint_url,
        api_key=config.api_key,
        model=config.model,
    )
    sub_request_context = build_request_context(
        endpoint_url=config.endpoint_url,
        api_key=config.api_key,
        model=config.sub_model,
    )
    state = create_chat_session(
        chat_request_context=chat_request_context,
        run_request_context=chat_request_context,
        sub_request_context=sub_request_context,
        attachments=attachments,
    )
    initial_prompt = config.initial_prompt or _read_non_interactive_prompt()
    if initial_prompt is not None:
        turn_result = execute_chat_turn(
            state,
            initial_prompt,
            on_run_start=print,
        )
        state = turn_result.state
        turn = turn_result.turn
        if config.verbose and turn.run_summary is not None:
            _print_run_summary(turn.run_summary)
        print(turn.assistant_text)
        if not sys.stdin.isatty():
            return 0

    if sys.stdin.isatty():
        print("Type /help for commands.")
    run_chat_session(state, verbose=config.verbose)
    return 0


def run_run_command(config: RunCLIConfig) -> int:
    attachments = convert_files_to_attachments(config.files)
    main_request_context = build_request_context(
        endpoint_url=config.endpoint_url,
        api_key=config.api_key,
        model=config.model,
    )
    sub_request_context = build_request_context(
        endpoint_url=config.endpoint_url,
        api_key=config.api_key,
        model=config.sub_model,
    )
    result = execute_repl_session(
        ReplExecutionRequest(
            prompt=build_run_prompt(config.prompt, attachments),
            setup=ReplSetupRequest(
                request_context=sub_request_context,
                context_payload=build_run_context_payload(attachments),
                files=[
                    ReplFileRef(
                        source_path=attachment.path,
                        target_name=attachment.name,
                    )
                    for attachment in attachments
                ],
                functions=select_function_collection(config.mode, attachments),
            ),
            session_request_context=main_request_context,
        )
    )
    summary = RunSummary(
        termination_reason=result.termination_reason.value,
        final_answer=result.final_answer,
        total_iterations=result.total_iterations,
        total_tokens=result.total_tokens,
        total_time_seconds=result.total_time_seconds,
    )
    _print_run_summary(summary)
    if result.final_answer is None:
        raise RuntimeError("Agent run completed without a final answer.")
    print(result.final_answer)
    return 0
