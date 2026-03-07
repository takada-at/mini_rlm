from enum import StrEnum
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict
from requests import Session


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class MessageContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


class MessageContent(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: List[MessageContentPart] | str
    name: str | None = None


class Endpoint(BaseModel):
    url: str
    method: str = "GET"
    headers: Dict[str, str] | None = None
    body: str | None = None


class RequestContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: Session
    endpoint: Endpoint
    kwargs: Dict[str, Any] | None = None
    messages: List[MessageContent] | None = None


class RequestStatus(StrEnum):
    IDLE = "idle"
    REQUESTING = "requesting"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RequestCommandType(StrEnum):
    REQUEST = "request"
    EXIT = "exit"


class RequestResultType(StrEnum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    INVALID_RESPONSE = "invalid_response"
    SKIPPED = "skipped"


class RetryPolicy(BaseModel):
    max_attempts: int
    initial_backoff_seconds: float
    backoff_multiplier: float
    max_backoff_seconds: float
    jitter_ratio: float
    retryable_status_codes: List[int]


class RequestPayload(BaseModel):
    url: str
    headers: Dict[str, str]
    body: Dict[str, Any]
    timeout_seconds: float


class RequestState(BaseModel):
    status: RequestStatus
    payload: RequestPayload
    retry_policy: RetryPolicy
    attempt_count: int = 0
    next_delay_seconds: float = 0.0
    last_error_type: RequestResultType | None = None
    last_error_message: str | None = None
    response_json: Dict[str, Any] | None = None


class RequestCommand(BaseModel):
    type: RequestCommandType
    payload: RequestPayload | None = None
    delay_seconds: float | None = None


class CommandResult(BaseModel):
    type: RequestResultType
    status_code: int | None = None
    response_json: Dict[str, Any] | None = None
    error_message: str | None = None


class APIRequestResult(BaseModel):
    response_json: Dict[str, Any]
    messages: List[MessageContent]
