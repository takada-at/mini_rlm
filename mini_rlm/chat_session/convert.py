import json
from pathlib import Path
from textwrap import dedent
from typing import Any

from mini_rlm.chat_session.data_model import (
    AttachmentKind,
    AttachmentRef,
    ChatDecision,
    ChatDecisionType,
    ChatSessionState,
)
from mini_rlm.llm import MessageContent, remove_think_tag_contents

_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def detect_attachment_kind(path: Path) -> AttachmentKind:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return AttachmentKind.PDF
    if suffix in _IMAGE_SUFFIXES:
        return AttachmentKind.IMAGE
    return AttachmentKind.OTHER


def convert_paths_to_attachments(paths: list[Path]) -> list[AttachmentRef]:
    return [
        AttachmentRef(path=path, name=path.name, kind=detect_attachment_kind(path))
        for path in paths
    ]


def create_chat_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "chat_decision_prompt.txt"
    with prompt_path.open("r", encoding="utf-8") as fp:
        return fp.read()


def build_attachment_summary(attachments: list[AttachmentRef]) -> str:
    if not attachments:
        return "- (none)"
    return "\n".join(
        f"- {attachment.name} ({attachment.kind.value})" for attachment in attachments
    )


def build_decision_messages(state: ChatSessionState) -> list[MessageContent]:
    if state.pending_user_text is None:
        raise ValueError("pending_user_text is required to build decision messages.")

    messages = [
        MessageContent(role="system", content=create_chat_system_prompt()),
    ]
    for turn in state.turns:
        messages.append(MessageContent(role="user", content=turn.user_text))
        messages.append(MessageContent(role="assistant", content=turn.assistant_text))

    messages.append(
        MessageContent(
            role="user",
            content=dedent(
                f"""\
                Attached files currently available:
                {build_attachment_summary(state.attachments)}

                User message:
                {state.pending_user_text}
                """
            ).strip(),
        )
    )
    return messages


def _extract_json_candidate(text: str) -> str:
    cleaned = remove_think_tag_contents(text).strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].lstrip()
    return cleaned


def _decode_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate = _extract_json_candidate(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise ValueError("chat decision response does not contain a JSON object.")
        parsed, _ = decoder.raw_decode(candidate[start:])

    if not isinstance(parsed, dict):
        raise ValueError("chat decision response must be a JSON object.")
    return parsed


def parse_chat_decision(text: str) -> ChatDecision:
    parsed = _decode_json_object(text)
    return ChatDecision.model_validate(parsed)


def validate_chat_decision(
    decision: ChatDecision,
    attachments: list[AttachmentRef],
) -> ChatDecision:
    attachment_names = {attachment.name for attachment in attachments}
    normalized_file_names = list(dict.fromkeys(decision.file_names))
    if any(file_name not in attachment_names for file_name in normalized_file_names):
        missing_files = [
            file_name
            for file_name in normalized_file_names
            if file_name not in attachment_names
        ]
        raise ValueError(f"run_agent requested unknown files: {missing_files}")

    normalized = decision.model_copy(update={"file_names": normalized_file_names})
    if normalized.type == ChatDecisionType.RESPOND_CHAT:
        if normalized.message is None or normalized.message.strip() == "":
            raise ValueError("respond_chat requires a non-empty message.")
        return normalized

    required_fields = {
        "task": normalized.task,
        "reason": normalized.reason,
        "success_criteria": normalized.success_criteria,
        "user_facing_preamble": normalized.user_facing_preamble,
    }
    missing_fields = [
        field_name
        for field_name, field_value in required_fields.items()
        if field_value is None or field_value.strip() == ""
    ]
    if missing_fields:
        raise ValueError(f"run_agent is missing required fields: {missing_fields}")
    return normalized


def build_forced_run_decision(
    user_text: str,
    attachments: list[AttachmentRef],
) -> ChatDecision:
    return ChatDecision(
        type=ChatDecisionType.RUN_AGENT,
        task=user_text,
        reason="The user explicitly requested an agent run.",
        file_names=[attachment.name for attachment in attachments],
        success_criteria="Return the best possible answer to the user's request.",
        user_facing_preamble="Running the agent on the current request.",
    )


def build_run_context_payload(attachments: list[AttachmentRef]) -> dict[str, object]:
    payload: dict[str, object] = {
        "attached_files": [attachment.name for attachment in attachments],
        "note": "The listed files have already been added to the REPL working directory.",
    }
    pdf_files = [
        attachment.name
        for attachment in attachments
        if attachment.kind == AttachmentKind.PDF
    ]
    image_files = [
        attachment.name
        for attachment in attachments
        if attachment.kind == AttachmentKind.IMAGE
    ]
    if len(pdf_files) == 1:
        payload["pdf_path"] = pdf_files[0]
    if len(image_files) == 1:
        payload["image_path"] = image_files[0]
    return payload


def build_run_prompt(
    state: ChatSessionState,
    decision: ChatDecision,
    max_history_turns: int = 4,
) -> str:
    if state.pending_user_text is None:
        raise ValueError("pending_user_text is required to build the run prompt.")

    recent_turns = state.turns[-max_history_turns:]
    history_lines = []
    for turn in recent_turns:
        history_lines.append(f"User: {turn.user_text}")
        history_lines.append(f"Assistant: {turn.assistant_text}")
    history_block = "\n".join(history_lines) if history_lines else "(none)"

    selected_files = "\n".join(f"- {file_name}" for file_name in decision.file_names)
    if selected_files == "":
        selected_files = "- (none)"

    return dedent(
        f"""\
        You are running inside mini_rlm agent mode.
        The selected files are already available in the REPL working directory.

        Recent chat history:
        {history_block}

        Current user message:
        {state.pending_user_text}

        Task:
        {decision.task}

        Success criteria:
        {decision.success_criteria}

        Selected files:
        {selected_files}

        Finish by calling FINAL(...) or FINAL_VAR(...).
        Return a concise final answer.
        """
    ).strip()
