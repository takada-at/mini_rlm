import os
from pathlib import Path

from mini_rlm.chat_session import (
    AttachmentKind,
    AttachmentRef,
    RunSummary,
    build_attachment_summary,
    convert_paths_to_attachments,
)
from mini_rlm.cli.data_model import ChatCLIInput, ChatCLIInputType, RunMode
from mini_rlm.custom_functions import (
    image_function_collection,
    merge_function_collections,
    minimal_function_collection,
    pdf_function_collection,
)
from mini_rlm.llm import RequestContext, create_request_context

ENDPOINT_ENV = "API_ENDPOINT"
API_KEY_ENV = "API_KEY"
MODEL = "openai/gpt-5.3-codex"
SUB_MODEL = "qwen/qwen3.5-35b-a3b"


def require_env(name: str, value: str | None = None) -> str:
    resolved = (
        value if value is not None and value.strip() != "" else os.environ.get(name)
    )
    if resolved is None or resolved.strip() == "":
        raise RuntimeError(f"Environment variable {name} is required.")
    return resolved


def resolve_file_paths(file_values: list[str | Path] | None) -> list[Path]:
    resolved_paths: list[Path] = []
    for file_value in file_values or []:
        path = (
            file_value.expanduser()
            if isinstance(file_value, Path)
            else Path(file_value).expanduser()
        )
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        resolved_paths.append(path)
    return resolved_paths


def _normalize_chat_file_path(file_path_text: str) -> Path:
    normalized_text = file_path_text.strip()
    if normalized_text == "":
        raise ValueError("Usage: /add <path>")
    if (
        len(normalized_text) >= 2
        and normalized_text[0] == normalized_text[-1]
        and normalized_text[0] in {"'", '"'}
    ):
        normalized_text = normalized_text[1:-1].strip()
    if normalized_text == "":
        raise ValueError("Usage: /add <path>")
    return Path(normalized_text).expanduser()


def parse_chat_input(user_text: str) -> ChatCLIInput:
    stripped_user_text = user_text.strip()
    if stripped_user_text == "":
        return ChatCLIInput(type=ChatCLIInputType.EMPTY)
    if stripped_user_text == "/exit":
        return ChatCLIInput(type=ChatCLIInputType.EXIT)
    if stripped_user_text == "/help":
        return ChatCLIInput(type=ChatCLIInputType.HELP)
    if stripped_user_text == "/files":
        return ChatCLIInput(type=ChatCLIInputType.FILES)
    if stripped_user_text == "/reset":
        return ChatCLIInput(type=ChatCLIInputType.RESET)
    if stripped_user_text == "/add":
        return ChatCLIInput(
            type=ChatCLIInputType.INVALID,
            error_message="Usage: /add <path>",
        )
    if stripped_user_text.startswith("/add "):
        file_path = stripped_user_text[5:].strip()
        if file_path == "":
            return ChatCLIInput(
                type=ChatCLIInputType.INVALID,
                error_message="Usage: /add <path>",
            )
        try:
            normalized_file_path = _normalize_chat_file_path(file_path)
        except ValueError as error:
            return ChatCLIInput(
                type=ChatCLIInputType.INVALID,
                error_message=str(error),
            )
        return ChatCLIInput(
            type=ChatCLIInputType.ADD_FILE,
            file_path=normalized_file_path,
        )
    if stripped_user_text == "/run":
        return ChatCLIInput(
            type=ChatCLIInputType.INVALID,
            error_message="Usage: /run <prompt>",
        )
    if stripped_user_text.startswith("/run "):
        message = stripped_user_text[5:].strip()
        if message == "":
            return ChatCLIInput(
                type=ChatCLIInputType.INVALID,
                error_message="Usage: /run <prompt>",
            )
        return ChatCLIInput(
            type=ChatCLIInputType.SEND_MESSAGE,
            message=message,
            force_run=True,
        )
    return ChatCLIInput(
        type=ChatCLIInputType.SEND_MESSAGE,
        message=stripped_user_text,
    )


def build_request_context(
    endpoint_url: str,
    api_key: str,
    model: str,
) -> RequestContext:
    return create_request_context(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model=model,
    )


def resolve_run_mode(
    mode: RunMode,
    attachments: list[AttachmentRef],
) -> RunMode:
    if mode != RunMode.AUTO:
        return mode
    has_pdf = any(attachment.kind == AttachmentKind.PDF for attachment in attachments)
    has_image = any(
        attachment.kind == AttachmentKind.IMAGE for attachment in attachments
    )
    if has_pdf and has_image:
        return RunMode.AUTO
    if has_pdf:
        return RunMode.PDF
    if has_image:
        return RunMode.IMAGE
    return RunMode.MINIMAL


def select_function_collection(
    mode: RunMode,
    attachments: list[AttachmentRef],
):
    resolved_mode = resolve_run_mode(mode, attachments)
    if resolved_mode == RunMode.AUTO:
        return merge_function_collections(
            pdf_function_collection(),
            image_function_collection(),
        )
    if resolved_mode == RunMode.PDF:
        return pdf_function_collection()
    if resolved_mode == RunMode.IMAGE:
        return image_function_collection()
    return minimal_function_collection()


def convert_files_to_attachments(paths: list[Path]) -> list[AttachmentRef]:
    return convert_paths_to_attachments(paths)


def format_attachment_list(attachments: list[AttachmentRef]) -> str:
    return build_attachment_summary(attachments)


def build_run_prompt(
    prompt: str,
    attachments: list[AttachmentRef],
) -> str:
    if not attachments:
        return prompt
    return (
        f"{prompt}\n\n"
        "Attached files already available in the REPL working directory:\n"
        f"{format_attachment_list(attachments)}"
    )


def format_run_summary(summary: RunSummary) -> str:
    return (
        f"termination_reason: {summary.termination_reason}\n"
        f"total_iterations: {summary.total_iterations}\n"
        f"total_tokens: {summary.total_tokens}\n"
        f"total_time_seconds: {summary.total_time_seconds:.2f}"
    )
