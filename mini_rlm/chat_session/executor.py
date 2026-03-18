from collections.abc import Callable
from pathlib import Path

from mini_rlm.chat_session.convert import (
    build_decision_messages,
    build_forced_run_decision,
    build_run_context_payload,
    build_run_prompt,
    convert_paths_to_attachments,
    parse_chat_decision,
    validate_chat_decision,
)
from mini_rlm.chat_session.data_model import (
    AttachmentRef,
    ChatSessionCommandType,
    ChatSessionResultType,
    ChatSessionState,
    ChatSessionStatus,
    ChatTurnResult,
    CommandResult,
    RunSummary,
)
from mini_rlm.chat_session.reducer import reduce_chat_session
from mini_rlm.custom_functions import (
    image_function_collection,
    merge_function_collections,
    minimal_function_collection,
    pdf_function_collection,
)
from mini_rlm.llm import (
    RequestContext,
    convert_messages_str,
    get_detailed_token_usage_from_response,
    make_api_request,
)
from mini_rlm.repl_session import (
    ReplExecutionRequest,
    ReplSessionLimits,
    execute_repl_session,
)
from mini_rlm.repl_setup import ReplFileRef, ReplSetupRequest


def create_chat_session(
    chat_request_context: RequestContext,
    run_request_context: RequestContext | None = None,
    attachments: list[AttachmentRef] | None = None,
    run_limits: ReplSessionLimits | None = None,
) -> ChatSessionState:
    return ChatSessionState(
        chat_request_context=chat_request_context,
        run_request_context=run_request_context or chat_request_context,
        attachments=list(attachments or []),
        run_limits=run_limits,
    )


def add_attachment(state: ChatSessionState, file_path: Path) -> ChatSessionState:
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    attachment_paths = [attachment.path for attachment in state.attachments]
    attachment_paths = [path for path in attachment_paths if path != file_path]
    attachment_paths.append(file_path)
    attachments = convert_paths_to_attachments(attachment_paths)
    return state.model_copy(update={"attachments": attachments})


def reset_chat_session(state: ChatSessionState) -> ChatSessionState:
    return state.model_copy(
        update={
            "pending_user_text": None,
            "pending_decision": None,
            "status": ChatSessionStatus.IDLE,
            "turns": [],
            "last_error": None,
            "total_tokens": 0,
            "model_token_usages": [],
        }
    )


def _execute_decide_command(state: ChatSessionState) -> CommandResult:
    try:
        messages = build_decision_messages(state)
        response = make_api_request(state.chat_request_context, messages)
        if not response.messages:
            return CommandResult(
                command_type=ChatSessionCommandType.DECIDE,
                type=ChatSessionResultType.ERROR,
                error_message="The chat model returned no message.",
            )
        response_text = convert_messages_str(response.messages).strip()
        decision = validate_chat_decision(
            parse_chat_decision(response_text),
            state.attachments,
        )
        token_usage = get_detailed_token_usage_from_response(response)
        return CommandResult(
            command_type=ChatSessionCommandType.DECIDE,
            type=ChatSessionResultType.SUCCESS,
            decision=decision,
            consumed_tokens=token_usage.total_tokens,
            model_token_usages=token_usage.model_token_usages,
        )
    except Exception as error:
        return CommandResult(
            command_type=ChatSessionCommandType.DECIDE,
            type=ChatSessionResultType.ERROR,
            error_message=(
                f"The chat model returned an invalid decision. Details: {error}"
            ),
        )


def _resolve_selected_attachments(
    state: ChatSessionState,
) -> list[AttachmentRef]:
    decision = state.pending_decision
    if decision is None:
        raise ValueError("pending_decision is required before running the agent.")
    if not decision.file_names:
        return []
    by_name: dict[str, AttachmentRef] = {}
    for attachment in state.attachments:
        if attachment.name in by_name:
            raise ValueError(f"Duplicate attachment name: {attachment.name}")
        by_name[attachment.name] = attachment
    return [by_name[file_name] for file_name in decision.file_names]


def _select_function_collection(attachments: list[AttachmentRef]):
    has_pdf = any(attachment.kind.value == "pdf" for attachment in attachments)
    has_image = any(attachment.kind.value == "image" for attachment in attachments)
    if has_pdf and has_image:
        return merge_function_collections(
            pdf_function_collection(),
            image_function_collection(),
        )
    if has_pdf:
        return pdf_function_collection()
    if has_image:
        return image_function_collection()
    return minimal_function_collection()


def _execute_run_agent_command(state: ChatSessionState) -> CommandResult:
    try:
        decision = state.pending_decision
        if decision is None:
            raise ValueError("pending_decision is required before running the agent.")
        selected_attachments = _resolve_selected_attachments(state)
        result = execute_repl_session(
            ReplExecutionRequest(
                prompt=build_run_prompt(state, decision),
                setup=ReplSetupRequest(
                    request_context=state.run_request_context,
                    context_payload=build_run_context_payload(selected_attachments),
                    files=[
                        ReplFileRef(
                            source_path=attachment.path,
                            target_name=attachment.name,
                        )
                        for attachment in selected_attachments
                    ],
                    functions=_select_function_collection(selected_attachments),
                ),
                limits=state.run_limits,
                session_request_context=state.run_request_context,
            )
        )
        run_summary = RunSummary(
            termination_reason=result.termination_reason.value,
            final_answer=result.final_answer,
            total_iterations=result.total_iterations,
            total_tokens=result.total_tokens,
            total_time_seconds=result.total_time_seconds,
        )
        assistant_text = result.final_answer
        if assistant_text is None or assistant_text.strip() == "":
            assistant_text = (
                "The agent run finished without a final answer. "
                f"termination_reason={result.termination_reason.value}"
            )
        return CommandResult(
            command_type=ChatSessionCommandType.RUN_AGENT,
            type=ChatSessionResultType.SUCCESS,
            assistant_text=assistant_text,
            run_summary=run_summary,
            consumed_tokens=result.total_tokens,
            model_token_usages=result.model_token_usages,
        )
    except Exception as error:
        return CommandResult(
            command_type=ChatSessionCommandType.RUN_AGENT,
            type=ChatSessionResultType.ERROR,
            error_message=f"The agent run failed: {error}",
        )


def execute_chat_turn(
    state: ChatSessionState,
    user_text: str,
    force_run: bool = False,
    on_run_start: Callable[[str], None] | None = None,
) -> ChatTurnResult:
    stripped_user_text = user_text.strip()
    if stripped_user_text == "":
        raise ValueError("user_text must not be empty.")

    working_state = state.model_copy(
        update={
            "pending_user_text": stripped_user_text,
            "pending_decision": (
                build_forced_run_decision(stripped_user_text, state.attachments)
                if force_run
                else None
            ),
        }
    )
    prev_result: CommandResult | None = None
    has_announced_run = False
    while True:
        working_state, command = reduce_chat_session(working_state, prev_result)
        if command.type == ChatSessionCommandType.COMPLETE_TURN:
            if not working_state.turns:
                raise RuntimeError("chat turn finished without producing a turn.")
            return ChatTurnResult(state=working_state, turn=working_state.turns[-1])
        if command.type == ChatSessionCommandType.DECIDE:
            prev_result = _execute_decide_command(working_state)
            continue
        if command.type == ChatSessionCommandType.RUN_AGENT:
            if (
                not has_announced_run
                and on_run_start is not None
                and working_state.pending_decision is not None
                and working_state.pending_decision.user_facing_preamble is not None
            ):
                on_run_start(working_state.pending_decision.user_facing_preamble)
                has_announced_run = True
            prev_result = _execute_run_agent_command(working_state)
            continue
        raise RuntimeError(f"Unsupported chat session command: {command.type}")


def run_chat_session(
    state: ChatSessionState,
    input_fn: Callable[[str], str] = input,
    write_fn: Callable[[str], None] = print,
    verbose: bool = False,
) -> ChatSessionState:
    current_state = state
    while True:
        try:
            user_text = input_fn("mini-rlm> ").strip()
        except EOFError:
            write_fn("")
            return current_state

        if user_text == "":
            continue
        if user_text == "/exit":
            return current_state
        if user_text == "/help":
            write_fn(
                "Commands: /help, /files, /add <path>, /reset, /run <prompt>, /exit"
            )
            continue
        if user_text == "/files":
            if not current_state.attachments:
                write_fn("(no attached files)")
            else:
                for attachment in current_state.attachments:
                    write_fn(f"{attachment.name} ({attachment.kind.value})")
            continue
        if user_text.startswith("/add "):
            file_path = Path(user_text[5:].strip()).expanduser()
            try:
                current_state = add_attachment(current_state, file_path)
            except Exception as error:
                write_fn(f"Failed to attach file: {error}")
                continue
            write_fn(f"Attached: {file_path.name}")
            continue
        if user_text == "/reset":
            current_state = reset_chat_session(current_state)
            write_fn("Chat session reset.")
            continue
        force_run = False
        if user_text == "/run":
            write_fn("Usage: /run <prompt>")
            continue
        if user_text.startswith("/run "):
            user_text = user_text[5:].strip()
            force_run = True
        try:
            turn_result = execute_chat_turn(
                current_state,
                user_text,
                force_run=force_run,
                on_run_start=write_fn,
            )
        except Exception as error:
            write_fn(f"Failed to execute chat turn: {error}")
            continue
        current_state = turn_result.state
        turn = turn_result.turn
        if verbose and turn.run_summary is not None:
            write_fn(
                "run summary: "
                f"termination_reason={turn.run_summary.termination_reason} "
                f"iterations={turn.run_summary.total_iterations} "
                f"tokens={turn.run_summary.total_tokens} "
                f"elapsed={turn.run_summary.total_time_seconds:.2f}s"
            )
        write_fn(turn.assistant_text)
