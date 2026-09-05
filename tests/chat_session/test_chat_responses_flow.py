from typing import Any
from unittest.mock import Mock

import pytest
import requests

from mini_rlm.chat_session import create_chat_session, execute_chat_turn
from mini_rlm.llm import create_request_context


def _response(text: str, identifier: str) -> tuple[Mock, list[dict[str, Any]]]:
    output: list[dict[str, Any]] = [
        {
            "type": "reasoning",
            "id": f"rs_{identifier}",
            "summary": [],
            "encrypted_content": f"opaque_{identifier}",
        },
        {
            "type": "message",
            "id": f"msg_{identifier}",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        },
    ]
    response = Mock()
    response.json.return_value = {
        "status": "completed",
        "model": "resolved",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    return response, output


def test_chat_replays_received_items_after_transport_retry_and_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # テストしたいふるまい: HTTP境界から次ターンまで推論を保持し、通信再試行や判断JSONエラーで重複しない。
    # give: timeout後に正常応答、その後JSONエラー、最後に正常応答を返すHTTP境界
    context = create_request_context(
        "https://example.invalid/v1/responses", "test", api_type="responses"
    )
    first, first_items = _response('{"type":"respond_chat","message":"Hi"}', "first")
    invalid, invalid_items = _response("not a decision", "invalid")
    last, _ = _response('{"type":"respond_chat","message":"Recovered"}', "last")
    request = Mock(side_effect=[requests.Timeout("timeout"), first, invalid, last])
    monkeypatch.setattr(context.session, "request", request)
    monkeypatch.setattr("mini_rlm.llm.api_request.time.sleep", lambda _: None)
    # when: 実際のAPI変換・リトライ・チャットreducerを通して3ターン実行する
    one = execute_chat_turn(create_chat_session(context), "hello")
    two = execute_chat_turn(one.state, "next")
    three = execute_chat_turn(two.state, "retry")
    # then: 次回ペイロードに元の全項目が一度だけ現れ、失敗時も受信内容とusageを保持する
    assert one.turn.assistant_text == "Hi"
    assert "invalid decision" in two.turn.assistant_text
    assert three.turn.assistant_text == "Recovered"
    assert (
        request.call_args_list[0].kwargs["json"]
        == request.call_args_list[1].kwargs["json"]
    )
    second_payload = request.call_args_list[2].kwargs["json"]
    third_payload = request.call_args_list[3].kwargs["json"]
    assert second_payload["input"][2:4] == first_items
    assert third_payload["input"][5:7] == invalid_items
    assert len(third_payload["input"]) == 9
    assert three.state.total_tokens == 45
    assert three.state.model_token_usages[0].completion_tokens == 15
    assert context.messages is None
