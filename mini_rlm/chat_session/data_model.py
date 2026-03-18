from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mini_rlm.llm import ModelTokenUsage, RequestContext
from mini_rlm.repl_session import ReplSessionLimits


class AttachmentKind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    OTHER = "other"


class ChatDecisionType(StrEnum):
    RESPOND_CHAT = "respond_chat"
    RUN_AGENT = "run_agent"


class ChatSessionStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"


class ChatSessionCommandType(StrEnum):
    DECIDE = "decide"
    RUN_AGENT = "run_agent"
    COMPLETE_TURN = "complete_turn"


class ChatSessionResultType(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class AttachmentRef(BaseModel):
    path: Path
    name: str
    kind: AttachmentKind


class RunSummary(BaseModel):
    termination_reason: str
    final_answer: str | None
    total_iterations: int
    total_tokens: int
    total_time_seconds: float


class ChatDecision(BaseModel):
    type: ChatDecisionType
    message: str | None = None
    task: str | None = None
    reason: str | None = None
    file_names: list[str] = Field(default_factory=list)
    success_criteria: str | None = None
    user_facing_preamble: str | None = None


class ChatTurn(BaseModel):
    user_text: str
    assistant_text: str
    decision_type: ChatDecisionType
    selected_files: list[str] = Field(default_factory=list)
    reason: str | None = None
    run_summary: RunSummary | None = None


class ChatSessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat_request_context: RequestContext
    run_request_context: RequestContext
    sub_request_context: RequestContext | None = None
    attachments: list[AttachmentRef] = Field(default_factory=list)
    turns: list[ChatTurn] = Field(default_factory=list)
    run_limits: ReplSessionLimits | None = None
    status: ChatSessionStatus = ChatSessionStatus.IDLE
    pending_user_text: str | None = None
    pending_decision: ChatDecision | None = None
    total_tokens: int = 0
    model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
    last_error: str | None = None


class ChatSessionCommand(BaseModel):
    type: ChatSessionCommandType


class CommandResult(BaseModel):
    command_type: ChatSessionCommandType
    type: ChatSessionResultType
    decision: ChatDecision | None = None
    assistant_text: str | None = None
    run_summary: RunSummary | None = None
    consumed_tokens: int = 0
    model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
    error_message: str | None = None


class ChatTurnResult(BaseModel):
    state: ChatSessionState
    turn: ChatTurn
