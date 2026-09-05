from enum import StrEnum
from typing import List

from pydantic import BaseModel, ConfigDict, Field

from mini_rlm.llm import HistoryItem, ModelTokenUsage, RequestContext
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
    API_REQUEST_FAILED = "APIRequestFailed"
    CONTEXT_LIMIT_EXCEEDED = "ContextLimitExceeded"


class ReplSessionLimits(BaseModel):
    token_limit: int
    iteration_limit: int
    timeout_seconds: float
    error_threshold: int
    context_window_tokens: int = Field(default=128_000, gt=0)
    output_token_reserve: int = Field(default=4096, ge=0)
    image_token_estimate: int = Field(default=8192, gt=0)
    compacting_threshold_rate: float = Field(default=0.85, gt=0, le=1)


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
    retryable: bool = True
    consumed_tokens: int = 0
    unknown_usage_count: int = 0
    model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
    last_llm_message: str | None = None
    last_llm_items: list[HistoryItem] = Field(default_factory=list)
    repl_results: List[ReplSessionHistoryEntry] | None = None
    is_complete: bool | None = None
    new_messages: List[HistoryItem] | None = None
    compacted_messages: List[HistoryItem] | None = None
    history_includes_prompt: bool = False
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
    unknown_usage_count: int = 0
    model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
    current_history_tokens: int = 0
    input_prefix: list[HistoryItem] = Field(default_factory=list)
    error_count: int = 0
    is_complete: bool = False
    is_cancelled: bool = False
    last_llm_message: str | None = None
    last_llm_items: list[HistoryItem] = Field(default_factory=list)
    repl_results: List[ReplSessionHistoryEntry] | None = None
    last_command_type: ReplSessionCommandType | None = None
    termination_reason: TerminationReason | None = None
    messages: List[HistoryItem] | None = None
    history_includes_prompt: bool = False
    ended_at_seconds: float | None = None

    repl_history: List[ReplSessionHistoryEntry] | None = None
    # This is the full history of code executions and their results, used for final output and debugging.

    final_answer: str | None = None

    def is_token_limit_exceeded(self) -> bool:
        return self.total_tokens > self.limits.token_limit

    def is_compaction_limit_exceeded(self) -> bool:
        return (
            self.current_history_tokens
            > (self.limits.context_window_tokens - self.limits.output_token_reserve)
            * self.limits.compacting_threshold_rate
        )


class ReplSessionResult(BaseModel):
    termination_reason: TerminationReason
    final_answer: str | None
    total_iterations: int
    total_tokens: int
    unknown_usage_count: int = 0
    model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
    total_time_seconds: float
    repl_history: List[ReplSessionHistoryEntry] | None = None
