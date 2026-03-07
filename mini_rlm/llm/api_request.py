import random
import time
from typing import Any, Dict, List

from mini_rlm.debug_logger import get_logger
from mini_rlm.llm.data_model import (
    APIRequestResult,
    MessageContent,
    RequestContext,
    RequestPayload,
    RequestState,
    RequestStatus,
    RetryPolicy,
)
from mini_rlm.llm.executor import execute_request_loop


def dump_messages(messages: List[MessageContent]) -> List[Dict[str, Any]]:
    """Dump a list of MessageContent to a list of dicts for JSON serialization."""
    res = []
    for message in messages:
        content: str | List[Dict[str, Any]] = []
        if isinstance(message.content, str):
            content = message.content
        else:
            assert isinstance(content, list)
            for part in message.content:
                if part.type == "text":
                    content.append({"type": "text", "text": part.text})
                elif part.type == "image_url":
                    assert part.image_url is not None
                    content.append(
                        {"type": "image_url", "image_url": part.image_url.model_dump()}
                    )
                else:
                    raise ValueError(f"Unsupported message content part: {part}")
        res.append({"role": message.role, "content": content})
    return res


def make_api_request(
    context: RequestContext, messages: List[MessageContent]
) -> APIRequestResult:
    """Make an API request to the endpoint specified in *context* with the given *messages*."""
    final_state = run_api_request(context, messages)
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

    response_data = final_state.response_json
    if (
        "choices" in response_data
        and isinstance(response_data["choices"], list)
        and len(response_data["choices"]) > 0
    ):
        return APIRequestResult(
            response_json=response_data,
            messages=[
                MessageContent.model_validate(choice["message"])
                for choice in response_data["choices"]
            ],
        )
    else:
        return APIRequestResult(
            response_json=response_data,
            messages=[],
        )


def run_api_request(
    context: RequestContext, messages: List[MessageContent]
) -> RequestState:
    """Make an API request to the endpoint specified in *context* with the given *messages*."""
    dict_messages = dump_messages(messages)
    if context.messages:
        dict_messages = dump_messages(context.messages) + dict_messages
    request_body: Dict[str, Any] = {"messages": dict_messages}
    if context.kwargs:
        request_body.update(context.kwargs)

    payload = RequestPayload(
        url=context.endpoint.url,
        headers=context.endpoint.headers or {},
        body=request_body,
        timeout_seconds=30.0,
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
            len(request_payload.body.get("messages", [])),
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
