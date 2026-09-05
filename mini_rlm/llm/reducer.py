from mini_rlm.llm.data_model import (
    APIRequestResult,
    CommandResult,
    RequestCommand,
    RequestCommandType,
    RequestResultType,
    RequestState,
    RequestStatus,
    TokenUsage,
)
from mini_rlm.llm.token_usage import (
    get_detailed_token_usage_from_response,
    merge_model_token_usages,
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
    if result.type in (
        RequestResultType.TIMEOUT,
        RequestResultType.NETWORK_ERROR,
        RequestResultType.INVALID_RESPONSE,
    ):
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

    response_json = prev_command_result.response_json or {}
    model_name = response_json.get("model") or prev_state.payload.body.get("model")
    usage = get_detailed_token_usage_from_response(
        APIRequestResult(
            response_json=response_json,
            messages=[],
            resolved_model_name=model_name if isinstance(model_name, str) else None,
        )
    )
    prev_state = prev_state.model_copy(
        update={
            "response_json": prev_command_result.response_json,
            "token_usage": TokenUsage(
                total_tokens=prev_state.token_usage.total_tokens + usage.total_tokens,
                unknown_usage_count=prev_state.token_usage.unknown_usage_count
                + usage.unknown_usage_count,
                model_token_usages=merge_model_token_usages(
                    prev_state.token_usage.model_token_usages, usage.model_token_usages
                ),
            ),
        }
    )

    if prev_command_result.type == RequestResultType.SUCCESS:
        next_state = prev_state.model_copy(
            update={
                "status": RequestStatus.SUCCEEDED,
                "response_json": prev_command_result.response_json,
                "message": prev_command_result.message,
                "parsed_response": prev_command_result.parsed_response,
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
