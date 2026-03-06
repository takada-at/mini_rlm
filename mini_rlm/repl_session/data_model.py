from enum import StrEnum
from typing import List

from pydantic import BaseModel

from mini_rlm.llm.data_model import MessageContent, RequestContext
from mini_rlm.repl.data_model import ReplResult


class ReplSessionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplSessionCommandType(StrEnum):
    EXIT = "exit"
    COMPLETE = "complete"
    COMPACTING = "compacting"
    CALL_LLM = "call_llm"
    EXECUTE_CODE = "execute_code"
    APPEND_HISTORY = "append_history"
    CHECK_COMPLETE = "check_complete"


class ReplSessionResultType(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class TerminationReason(StrEnum):
    TOKEN_LIMIT_EXCEEDED = "TokenLimitExceeded"
    ITERATIONS_EXHAUSTED = "IterationsExhausted"
    TIMEOUT = "Timeout"
    ERROR_THRESHOLD_EXCEEDED = "ErrorThresholdExceeded"
    CANCELLED = "Cancelled"
    COMPLETED = "Completed"


class ReplSessionLimits(BaseModel):
    token_limit: int
    iteration_limit: int
    timeout_seconds: float
    error_threshold: int
    history_limit: int


class ReplSessionCommand(BaseModel):
    type: ReplSessionCommandType


class ReplSessionHistoryEntry(BaseModel):
    code: str
    repl_result: ReplResult | None = None


class CommandResult(BaseModel):
    command_type: ReplSessionCommandType
    type: ReplSessionResultType
    consumed_tokens: int = 0
    last_llm_message: str | None = None
    repl_results: List[ReplSessionHistoryEntry] | None = None
    is_complete: bool | None = None
    new_messages: List[MessageContent] | None = None
    compacted_messages: List[MessageContent] | None = None
    final_answer: str | None = None
    error_message: str | None = None


class ReplSessionState(BaseModel):
    status: ReplSessionStatus
    limits: ReplSessionLimits
    started_at_seconds: float
    current_time_seconds: float
    iteration_count: int = 0
    total_tokens: int = 0
    error_count: int = 0
    is_complete: bool = False
    is_cancelled: bool = False
    last_llm_message: str | None = None
    repl_results: List[ReplSessionHistoryEntry] | None = None
    last_command_type: ReplSessionCommandType | None = None
    termination_reason: TerminationReason | None = None
    messages: List[MessageContent] | None = None
    final_answer: str | None = None


class ReplSessionExecutorState(BaseModel):
    request_context: RequestContext
    limits: ReplSessionLimits
    system_prompt: str
    prompt: str
    messages: List[MessageContent] | None = None
