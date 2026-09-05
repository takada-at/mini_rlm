import random
import time
from collections.abc import Sequence
from typing import Any, Dict

from mini_rlm.debug_logger import get_logger
from mini_rlm.llm.data_model import (
    APIRequestResult,
    HistoryItem,
    RequestContext,
    RequestOperation,
    RequestPayload,
    RequestState,
    RequestStatus,
    RetryPolicy,
)
from mini_rlm.llm.executor import execute_request_loop
from mini_rlm.llm.protocol import build_request_payload
from mini_rlm.llm.protocol import dump_messages as dump_messages


def make_api_request(
    context: RequestContext,
    messages: Sequence[HistoryItem],
    *,
    operation: RequestOperation = "create",
    include_context_messages: bool = True,
) -> APIRequestResult:
    """Make an API request to the endpoint specified in *context* with the given *messages*."""
    final_state = run_api_request(
        context,
        messages,
        operation=operation,
        include_context_messages=include_context_messages,
    )
    if (
        final_state.status != RequestStatus.SUCCEEDED
        or final_state.response_json is None
    ):
        error_type = (
            final_state.last_error_type.value
            if final_state.last_error_type is not None
            else "unknown"
        )
        error_message = final_state.last_error_message or "request failed"
        raise RuntimeError(f"LLM API request failed: {error_type}: {error_message}")
    parsed = final_state.parsed_response
    return APIRequestResult(
        response_json=final_state.response_json,
        messages=parsed.messages if parsed is not None else [],
        output_items=parsed.output_items if parsed is not None else [],
        resolved_model_name=_resolve_model_name(final_state.response_json, context),
    )


def run_api_request(
    context: RequestContext,
    messages: Sequence[HistoryItem],
    *,
    operation: RequestOperation = "create",
    include_context_messages: bool = True,
) -> RequestState:
    """Make an API request to the endpoint specified in *context* with the given *messages*."""
    payload = build_request_payload(
        context,
        messages,
        operation=operation,
        include_context_messages=include_context_messages,
    )
    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_backoff_seconds=0.5,
        backoff_multiplier=2.0,
        max_backoff_seconds=8.0,
        jitter_ratio=0.2,
        retryable_status_codes=[429, 500, 502, 503, 504],
    )
    initial_state = RequestState(
        status=RequestStatus.IDLE,
        payload=payload,
        retry_policy=retry_policy,
    )
    logger = get_logger()

    def send_request(request_payload: RequestPayload) -> Dict[str, Any]:
        logger.debug(
            "Sending request to %s with %s message(s)",
            request_payload.url,
            len(
                request_payload.body.get(
                    "input", request_payload.body.get("messages", [])
                )
            ),
        )
        response = context.session.request(
            "POST",
            request_payload.url,
            headers=request_payload.headers,
            json=request_payload.body,
            timeout=request_payload.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    final_state = execute_request_loop(
        initial_state=initial_state,
        send_request=send_request,
        sleep_fn=time.sleep,
        random_fn=random.random,
    )
    return final_state


def _resolve_model_name(
    response_json: Dict[str, Any],
    context: RequestContext,
) -> str | None:
    response_model_name = response_json.get("model")
    if isinstance(response_model_name, str) and response_model_name != "":
        return response_model_name
    if context.kwargs is None:
        return None
    request_model_name = context.kwargs.get("model")
    if isinstance(request_model_name, str) and request_model_name != "":
        return request_model_name
    return None
