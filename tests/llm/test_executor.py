from typing import Any

import requests

from mini_rlm.llm.data_model import (
    RequestPayload,
    RequestState,
    RequestStatus,
    RetryPolicy,
)
from mini_rlm.llm.executor import execute_request_loop


def build_state(*, max_attempts: int = 3, jitter_ratio: float = 0.2) -> RequestState:
    return RequestState(
        status=RequestStatus.IDLE,
        payload=RequestPayload(
            url="https://example.com/v1/chat/completions",
            headers={"Authorization": "Bearer test"},
            body={"messages": [{"role": "user", "content": "hello"}]},
            timeout_seconds=5.0,
        ),
        retry_policy=RetryPolicy(
            max_attempts=max_attempts,
            initial_backoff_seconds=0.5,
            backoff_multiplier=2.0,
            max_backoff_seconds=8.0,
            jitter_ratio=jitter_ratio,
            retryable_status_codes=[429, 500, 502, 503, 504],
        ),
    )


def make_http_error(status_code: int, message: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(message, response=response)


def test_execute_request_loop_retries_timeout_then_succeeds() -> None:
    # give: 1回目timeout、2回目successの送信関数
    state = build_state()
    calls: list[int] = []
    sleep_calls: list[float] = []

    def send_request(_: RequestPayload) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("request timed out")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def sleep_fn(seconds: float) -> None:
        sleep_calls.append(seconds)

    # when: executorを実行する
    final_state = execute_request_loop(
        initial_state=state,
        send_request=send_request,
        sleep_fn=sleep_fn,
        random_fn=lambda: 0.0,
    )

    # then: リトライ後に成功終了する
    assert len(calls) == 2
    assert final_state.status == RequestStatus.SUCCEEDED
    assert sleep_calls == [0.4]


def test_execute_request_loop_fails_when_max_attempts_reached() -> None:
    # give: 常に500を返す送信関数
    state = build_state(max_attempts=3)

    def send_request(_: RequestPayload) -> dict[str, Any]:
        raise make_http_error(500, "server error")

    # when: executorを実行する
    final_state = execute_request_loop(
        initial_state=state,
        send_request=send_request,
        sleep_fn=lambda _: None,
        random_fn=lambda: 0.0,
    )

    # then: max_attemptsでfailed終了する
    assert final_state.status == RequestStatus.FAILED
    assert final_state.attempt_count == 3


def test_execute_request_loop_uses_jitter_for_retry_sleep() -> None:
    # give: timeout後にsuccessし、jitterを最大側に固定する
    state = build_state(jitter_ratio=0.2)
    sleep_calls: list[float] = []
    calls: list[int] = []

    def send_request(_: RequestPayload) -> dict[str, Any]:
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("timeout")
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def sleep_fn(seconds: float) -> None:
        sleep_calls.append(seconds)

    # when: executorを実行する
    _ = execute_request_loop(
        initial_state=state,
        send_request=send_request,
        sleep_fn=sleep_fn,
        random_fn=lambda: 1.0,
    )

    # then: retry待機にjitterが反映される
    assert sleep_calls == [0.6]
