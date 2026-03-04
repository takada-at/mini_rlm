from mini_rlm.llm.data_model import (
    CommandResult,
    RequestCommand,
    RequestCommandType,
    RequestResultType,
    RequestState,
    RequestStatus,
)


def _compute_retry_delay_seconds(prev_state: RequestState, next_attempt: int) -> float:
    if next_attempt <= 1:
        return 0.0

    policy = prev_state.retry_policy
    delay = policy.initial_backoff_seconds * (
        policy.backoff_multiplier ** (next_attempt - 2)
    )
    return min(delay, policy.max_backoff_seconds)


def _is_retryable(prev_state: RequestState, result: CommandResult) -> bool:
    if result.type in (RequestResultType.TIMEOUT, RequestResultType.NETWORK_ERROR):
        return True

    if result.type == RequestResultType.HTTP_ERROR and result.status_code is not None:
        return result.status_code in prev_state.retry_policy.retryable_status_codes

    return False


def reduce_request(
    prev_state: RequestState,
    prev_command_result: CommandResult | None,
) -> tuple[RequestState, RequestCommand]:
    if prev_command_result is None:
        next_state = prev_state.model_copy(
            update={
                "status": RequestStatus.REQUESTING,
                "attempt_count": 1,
                "next_delay_seconds": 0.0,
                "last_error_type": None,
                "last_error_message": None,
            }
        )
        return (
            next_state,
            RequestCommand(
                type=RequestCommandType.REQUEST,
                payload=prev_state.payload,
                delay_seconds=0.0,
            ),
        )

    if prev_command_result.type == RequestResultType.SUCCESS:
        next_state = prev_state.model_copy(
            update={
                "status": RequestStatus.SUCCEEDED,
                "response_json": prev_command_result.response_json,
                "last_error_type": None,
                "last_error_message": None,
            }
        )
        return (next_state, RequestCommand(type=RequestCommandType.EXIT))

    can_retry = (
        _is_retryable(prev_state, prev_command_result)
        and prev_state.attempt_count < prev_state.retry_policy.max_attempts
    )
    if can_retry:
        next_attempt = prev_state.attempt_count + 1
        next_delay_seconds = _compute_retry_delay_seconds(prev_state, next_attempt)
        next_state = prev_state.model_copy(
            update={
                "status": RequestStatus.RETRY_WAIT,
                "attempt_count": next_attempt,
                "next_delay_seconds": next_delay_seconds,
                "last_error_type": prev_command_result.type,
                "last_error_message": prev_command_result.error_message,
            }
        )
        return (
            next_state,
            RequestCommand(
                type=RequestCommandType.REQUEST,
                payload=prev_state.payload,
                delay_seconds=next_delay_seconds,
            ),
        )

    next_state = prev_state.model_copy(
        update={
            "status": RequestStatus.FAILED,
            "last_error_type": prev_command_result.type,
            "last_error_message": prev_command_result.error_message,
        }
    )
    return (next_state, RequestCommand(type=RequestCommandType.EXIT))
