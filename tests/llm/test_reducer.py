from mini_rlm.llm.data_model import (
    CommandResult,
    RequestCommandType,
    RequestPayload,
    RequestResultType,
    RequestState,
    RequestStatus,
    RetryPolicy,
)
from mini_rlm.llm.reducer import reduce_request


def build_state() -> RequestState:
    return RequestState(
        status=RequestStatus.IDLE,
        payload=RequestPayload(
            url="https://example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            body={"messages": [{"role": "user", "content": "hello"}]},
            timeout_seconds=5.0,
        ),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.5,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
            jitter_ratio=0.2,
            retryable_status_codes=[429, 500, 502, 503, 504],
        ),
    )


def test_reduce_request_initial_step_returns_request_command() -> None:
    # give: 初期stateと直前実行結果なし
    prev_state = build_state()
    # when: reducerを実行する
    next_state, command = reduce_request(prev_state, None)
    # then: 1回目のrequestコマンドが返る
    assert next_state.status == RequestStatus.REQUESTING
    assert next_state.attempt_count == 1
    assert command.type == RequestCommandType.REQUEST
    assert command.payload == prev_state.payload
    assert command.delay_seconds == 0.0


def test_reduce_request_success_returns_exit_with_succeeded_state() -> None:
    # give: request実行後のstateと成功結果
    prev_state = build_state().model_copy(
        update={"status": RequestStatus.REQUESTING, "attempt_count": 1}
    )
    prev_result = CommandResult(
        type=RequestResultType.SUCCESS,
        response_json={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        },
    )
    # when: reducerを実行する
    next_state, command = reduce_request(prev_state, prev_result)
    # then: succeededになり終了コマンドが返る
    assert next_state.status == RequestStatus.SUCCEEDED
    assert next_state.response_json == prev_result.response_json
    assert command.type == RequestCommandType.EXIT


def test_reduce_request_timeout_returns_retry_request_when_attempts_remain() -> None:
    # give: 1回目失敗(timeout)でリトライ余地がある
    prev_state = build_state().model_copy(
        update={"status": RequestStatus.REQUESTING, "attempt_count": 1}
    )
    prev_result = CommandResult(
        type=RequestResultType.TIMEOUT,
        error_message="request timed out",
    )
    # when: reducerを実行する
    next_state, command = reduce_request(prev_state, prev_result)
    # then: retry待機状態になり2回目requestコマンドが返る
    assert next_state.status == RequestStatus.RETRY_WAIT
    assert next_state.attempt_count == 2
    assert next_state.next_delay_seconds == 0.5
    assert command.type == RequestCommandType.REQUEST
    assert command.delay_seconds == 0.5


def test_reduce_request_http_400_returns_exit_with_failed_state() -> None:
    # give: 400エラー結果
    prev_state = build_state().model_copy(
        update={"status": RequestStatus.REQUESTING, "attempt_count": 1}
    )
    prev_result = CommandResult(
        type=RequestResultType.HTTP_ERROR,
        status_code=400,
        error_message="bad request",
    )
    # when: reducerを実行する
    next_state, command = reduce_request(prev_state, prev_result)
    # then: 非リトライでfailedになり終了コマンドが返る
    assert next_state.status == RequestStatus.FAILED
    assert next_state.last_error_type == RequestResultType.HTTP_ERROR
    assert command.type == RequestCommandType.EXIT


def test_reduce_request_reaches_max_attempts_and_exits_failed() -> None:
    # give: max_attempts到達済みでサーバーエラーが返る
    prev_state = build_state().model_copy(
        update={"status": RequestStatus.REQUESTING, "attempt_count": 3}
    )
    prev_result = CommandResult(
        type=RequestResultType.HTTP_ERROR,
        status_code=500,
        error_message="internal server error",
    )
    # when: reducerを実行する
    next_state, command = reduce_request(prev_state, prev_result)
    # then: 追加リトライせずfailedで終了する
    assert next_state.status == RequestStatus.FAILED
    assert next_state.attempt_count == 3
    assert command.type == RequestCommandType.EXIT
