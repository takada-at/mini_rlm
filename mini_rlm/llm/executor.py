from collections.abc import Callable
from typing import Any

import requests

from mini_rlm.llm.data_model import (
    CommandResult,
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
        return CommandResult(
            type=RequestResultType.SUCCESS, response_json=response_json
        )
    except requests.Timeout as error:
        return CommandResult(
            type=RequestResultType.TIMEOUT,
            error_message=_normalize_error_message(error),
        )
    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response is not None else None
        return CommandResult(
            type=RequestResultType.HTTP_ERROR,
            status_code=status_code,
            error_message=_normalize_error_message(error),
        )
    except requests.RequestException as error:
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
