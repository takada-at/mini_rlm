from typing import Any, cast
from unittest.mock import Mock

import pytest
import requests

from mini_rlm.llm.api_request import make_api_request
from mini_rlm.llm.data_model import Endpoint, MessageContent, RequestContext


def test_make_api_request_retries_timeout_and_returns_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # give: 1回目timeout、2回目成功レスポンスを返すsession
    session = cast(Any, requests.Session())
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}]
    }
    session.request = Mock(side_effect=[requests.Timeout("timeout"), success_response])
    context = RequestContext(
        session=session,
        endpoint=Endpoint(
            url="https://example.com/v1/chat/completions", headers={"X-Test": "1"}
        ),
        kwargs={"model": "gpt-test"},
    )
    messages = [MessageContent(role="user", content="hello")]
    monkeypatch.setattr("mini_rlm.llm.api_request.time.sleep", lambda _: None)
    monkeypatch.setattr("mini_rlm.llm.api_request.random.random", lambda: 0.0)

    # when: APIリクエスト関数を実行する
    result = make_api_request(context, messages)
    res_messages = result.messages

    # then: リトライ後の成功結果を返す
    assert len(res_messages) == 1
    assert res_messages[0].role == "assistant"
    assert session.request.call_count == 2


def test_make_api_request_raises_when_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # give: 常に500エラーを返すsession
    session = cast(Any, requests.Session())
    response = requests.Response()
    response.status_code = 500
    http_error = requests.HTTPError("server error", response=response)
    session.request = Mock(side_effect=http_error)
    context = RequestContext(
        session=session,
        endpoint=Endpoint(
            url="https://example.com/v1/chat/completions", headers={"X-Test": "1"}
        ),
        kwargs={"model": "gpt-test"},
    )
    messages = [MessageContent(role="user", content="hello")]
    monkeypatch.setattr("mini_rlm.llm.api_request.time.sleep", lambda _: None)
    monkeypatch.setattr("mini_rlm.llm.api_request.random.random", lambda: 0.0)

    # when / then: 試行上限到達で例外を送出する
    with pytest.raises(RuntimeError):
        _ = make_api_request(context, messages)

    assert session.request.call_count == 5


def test_make_api_request_prefers_response_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # give: response と request の両方に model 名がある
    session = cast(Any, requests.Session())
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "model": "gpt-response",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    session.request = Mock(return_value=success_response)
    context = RequestContext(
        session=session,
        endpoint=Endpoint(
            url="https://example.com/v1/chat/completions", headers={"X-Test": "1"}
        ),
        kwargs={"model": "gpt-request"},
    )
    messages = [MessageContent(role="user", content="hello")]
    monkeypatch.setattr("mini_rlm.llm.api_request.time.sleep", lambda _: None)
    monkeypatch.setattr("mini_rlm.llm.api_request.random.random", lambda: 0.0)

    # when: API リクエスト関数を実行する
    result = make_api_request(context, messages)

    # then: response の model 名が優先される
    assert result.resolved_model_name == "gpt-response"


def test_make_api_request_falls_back_to_request_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # give: response に model 名が無く、request 側には model 名がある
    session = cast(Any, requests.Session())
    success_response = Mock()
    success_response.raise_for_status.return_value = None
    success_response.json.return_value = {
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
    }
    session.request = Mock(return_value=success_response)
    context = RequestContext(
        session=session,
        endpoint=Endpoint(
            url="https://example.com/v1/chat/completions", headers={"X-Test": "1"}
        ),
        kwargs={"model": "gpt-request"},
    )
    messages = [MessageContent(role="user", content="hello")]
    monkeypatch.setattr("mini_rlm.llm.api_request.time.sleep", lambda _: None)
    monkeypatch.setattr("mini_rlm.llm.api_request.random.random", lambda: 0.0)

    # when: API リクエスト関数を実行する
    result = make_api_request(context, messages)

    # then: request の model 名で補完される
    assert result.resolved_model_name == "gpt-request"
