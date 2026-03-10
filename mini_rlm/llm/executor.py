import traceback
from collections.abc import Callable
from typing import Any

import requests
from pydantic import ValidationError

from mini_rlm.debug_logger import get_logger
from mini_rlm.llm.data_model import (
    CommandResult,
    MessageContent,
    RequestCommand,
    RequestCommandType,
    RequestPayload,
    RequestResultType,
    RequestState,
)
from mini_rlm.llm.reducer import reduce_request


def _normalize_error_message(error: Exception) -> str:
    return str(error) or error.__class__.__name__


def _run_request_command(
    command: RequestCommand,
    send_request: Callable[[RequestPayload], dict[str, Any]],
) -> CommandResult:
    logger = get_logger()
    logger.debug(f"Executing request command: {command.type}")
    if command.type != RequestCommandType.REQUEST or command.payload is None:
        return CommandResult(
            type=RequestResultType.SKIPPED, error_message="invalid command"
        )

    try:
        response_json = send_request(command.payload)
        if (
            "choices" not in response_json
            or not isinstance(response_json["choices"], list)
            or len(response_json["choices"]) == 0
        ):
            return CommandResult(
                type=RequestResultType.INVALID_RESPONSE,
                error_message="response JSON does not contain 'choices'",
            )
        try:
            message = MessageContent.model_validate(
                response_json["choices"][0]["message"]
            )
        except ValidationError as error:
            logger.warning(f"Failed to parse message content from response: {error}")
            message = None
            return CommandResult(
                type=RequestResultType.INVALID_RESPONSE,
                error_message=f"response JSON 'choices' has invalid format: {error}",
            )
        return CommandResult(
            type=RequestResultType.SUCCESS, response_json=response_json, message=message
        )
    except requests.Timeout as error:
        logger.warning(f"Request timed out: {error}")
        return CommandResult(
            type=RequestResultType.TIMEOUT,
            error_message=_normalize_error_message(error),
        )
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        logger.warning(f"HTTP error occurred: {error}, status_code: {status_code}")
        traceback_str = traceback.format_exc()
        logger.debug(f"Traceback for HTTP error: {traceback_str}")
        return CommandResult(
            type=RequestResultType.HTTP_ERROR,
            status_code=status_code,
            error_message=_normalize_error_message(error),
        )
    except requests.RequestException as error:
        traceback_str = traceback.format_exc()
        logger.error(f"Network error occurred: {error}, traceback: {traceback_str}")
        return CommandResult(
            type=RequestResultType.NETWORK_ERROR,
            error_message=_normalize_error_message(error),
        )


def _compute_jittered_delay_seconds(
    delay_seconds: float,
    jitter_ratio: float,
    random_fn: Callable[[], float],
) -> float:
    if delay_seconds <= 0:
        return 0.0
    if jitter_ratio <= 0:
        return delay_seconds

    random_value = min(1.0, max(0.0, random_fn()))
    jitter_scale = 1.0 + ((random_value * 2.0) - 1.0) * jitter_ratio
    return max(0.0, delay_seconds * jitter_scale)


def execute_request_loop(
    initial_state: RequestState,
    send_request: Callable[[RequestPayload], dict[str, Any]],
    sleep_fn: Callable[[float], None],
    random_fn: Callable[[], float],
) -> RequestState:
    state = initial_state
    prev_result: CommandResult | None = None

    while True:
        state, command = reduce_request(state, prev_result)

        if command.type == RequestCommandType.EXIT:
            return state

        delay_seconds = command.delay_seconds or 0.0
        wait_seconds = _compute_jittered_delay_seconds(
            delay_seconds,
            state.retry_policy.jitter_ratio,
            random_fn,
        )
        if wait_seconds > 0:
            sleep_fn(wait_seconds)

        prev_result = _run_request_command(command, send_request)
