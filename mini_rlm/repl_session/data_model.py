from enum import StrEnum
from typing import List

from pydantic import BaseModel, ConfigDict

from mini_rlm.llm import MessageContent, RequestContext
from mini_rlm.repl import ReplResult
from mini_rlm.repl_setup import ReplSetupRequest


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
    UNKNOWN = "Unknown"


class ReplSessionLimits(BaseModel):
    token_limit: int
    iteration_limit: int
    timeout_seconds: float
    error_threshold: int
    compacting_threshold_rate: float = 0.85


class ReplExecutionRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    setup: ReplSetupRequest
    limits: ReplSessionLimits | None = None
    session_request_context: RequestContext | None = None


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
    prompt: str
    status: ReplSessionStatus
    limits: ReplSessionLimits
    started_at_seconds: float
    current_time_seconds: float
    iteration_count: int = 0
    total_tokens: int = 0
    current_history_tokens: int = 0
    error_count: int = 0
    is_complete: bool = False
    is_cancelled: bool = False
    last_llm_message: str | None = None
    repl_results: List[ReplSessionHistoryEntry] | None = None
    last_command_type: ReplSessionCommandType | None = None
    termination_reason: TerminationReason | None = None
    messages: List[MessageContent] | None = None
    ended_at_seconds: float | None = None

    repl_history: List[ReplSessionHistoryEntry] | None = None
    # This is the full history of code executions and their results, used for final output and debugging.

    final_answer: str | None = None

    def is_token_limit_exceeded(self) -> bool:
        return self.total_tokens > self.limits.token_limit

    def is_compaction_limit_exceeded(self) -> bool:
        return (
            self.current_history_tokens
            > self.limits.token_limit * self.limits.compacting_threshold_rate
        )


class ReplSessionResult(BaseModel):
    termination_reason: TerminationReason
    final_answer: str | None
    total_iterations: int
    total_tokens: int
    total_time_seconds: float
    repl_history: List[ReplSessionHistoryEntry] | None = None
