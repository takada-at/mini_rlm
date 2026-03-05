from enum import StrEnum

from pydantic import BaseModel


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


class ReplSessionState(BaseModel):
    status: ReplSessionStatus
    limits: ReplSessionLimits
    started_at_seconds: float
    current_time_seconds: float
    iteration_count: int = 0
    total_tokens: int = 0
    history_length: int = 0
    error_count: int = 0
    is_complete: bool = False
    is_cancelled: bool = False
    last_command_type: ReplSessionCommandType | None = None
    termination_reason: TerminationReason | None = None


class ReplSessionCommand(BaseModel):
    type: ReplSessionCommandType


class CommandResult(BaseModel):
    command_type: ReplSessionCommandType
    type: ReplSessionResultType
    consumed_tokens: int = 0
    history_length_delta: int = 0
    history_length_override: int | None = None
    is_complete: bool | None = None
    error_message: str | None = None
