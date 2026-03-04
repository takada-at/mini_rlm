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

    # then: リトライ後の成功結果を返す
    assert len(result) == 1
    assert result[0].role == "assistant"
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
