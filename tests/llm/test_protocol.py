from copy import deepcopy
from typing import Any

import pytest

from mini_rlm.llm import (
    APIRequestResult,
    HistoryItem,
    ImageURL,
    MessageContent,
    MessageContentPart,
    create_request_context,
    get_detailed_token_usage_from_response,
)
from mini_rlm.llm.convert import convert_messages_str
from mini_rlm.llm.protocol import (
    build_request_payload,
    dump_response_input,
    parse_response,
)


def _output() -> list[dict[str, Any]]:
    return [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "private summary"}],
            "encrypted_content": "opaque-reasoning",
            "future_field": {"preserve": True},
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "final_answer",
            "content": [
                {"type": "output_text", "text": "hello", "annotations": []},
                {"type": "output_text", "text": " world", "annotations": []},
            ],
        },
        {
            "type": "message",
            "id": "msg_2",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "!", "annotations": []}],
        },
    ]


def test_responses_replay_keeps_all_fields_and_extracts_only_visible_text() -> None:
    # テストしたいふるまい: 推論と応答の全項目を再送し、本文には推論を混ぜない。
    # give: 推論・複数メッセージ・未知フィールドを持つ完了応答
    output = _output()
    original = deepcopy(output)
    context = create_request_context(
        "https://example.invalid/v1/responses", "test", api_type="responses"
    )
    # when: 応答を解釈し、次の質問を追加してリクエストへ変換する
    parsed = parse_response({"status": "completed", "output": output}, "responses")
    history: list[HistoryItem] = [
        *parsed.output_items,
        MessageContent(role="user", content="next"),
    ]
    payload = build_request_payload(context, history)
    # then: 順序・ID・暗号化内容・付加情報が失われず、表示は本文だけになる
    assert payload.body["input"][:-1] == original
    assert convert_messages_str(parsed.messages) == "hello world!"
    assert output == original
    assert payload.body["store"] is False
    assert payload.body["include"] == ["reasoning.encrypted_content"]
    assert "messages" not in payload.body
    assert context.messages is None


def test_text_and_image_input_are_converted_without_mutating_configuration() -> None:
    # テストしたいふるまい: 画像のdata URLとdetailを維持し、includeを破壊せず推論保持を有効化する。
    # give: system指示、テキストと画像、呼び出し固有の設定
    context = create_request_context(
        "https://example.invalid/v1/responses",
        "test",
        api_type="responses",
        request_params={
            "include": ["message.output_text.logprobs"],
            "store": True,
            "reasoning": {"effort": "high", "context": "all_turns"},
        },
    )
    context.messages = [MessageContent(role="system", content="instructions")]
    original_params = deepcopy(context.kwargs)
    message = MessageContent(
        role="user",
        content=[
            MessageContentPart(type="text", text="Inspect"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURL(url="data:image/png;base64,abc", detail="high"),
            ),
        ],
    )
    # when: リクエストを組み立てる
    payload = build_request_payload(context, [message])
    # then: 画像形式と指示が変換され、元の設定は変更されない
    assert payload.body["input"] == [
        {"role": "system", "content": [{"type": "input_text", "text": "instructions"}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,abc",
                    "detail": "high",
                },
            ],
        },
    ]
    assert payload.body["store"] is False
    assert payload.body["reasoning"] == {"effort": "high", "context": "all_turns"}
    assert payload.body["include"] == [
        "message.output_text.logprobs",
        "reasoning.encrypted_content",
    ]
    assert context.kwargs == original_params


def test_compaction_output_is_replayed_as_a_complete_window() -> None:
    # テストしたいふるまい: compact専用ペイロードを使い、ユーザー入力と暗号化圧縮項目を丸ごと保持する。
    # give: 通常生成の設定とcompact応答
    context = create_request_context(
        "https://example.invalid/v1/responses/?api-version=test",
        "test",
        api_type="responses",
        request_params={
            "reasoning": {"effort": "high"},
            "max_output_tokens": 100,
            "instructions": "Keep the task",
        },
    )
    output = [
        {
            "type": "message",
            "id": "user_1",
            "role": "user",
            "content": [{"type": "input_text", "text": "task"}],
        },
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque-compaction"},
    ]
    # when: 圧縮用リクエストと次回の通常リクエストを作る
    payload = build_request_payload(
        context, [MessageContent(role="user", content="task")], operation="compact"
    )
    parsed = parse_response(
        {"object": "response.compaction", "output": output}, "responses", "compact"
    )
    replay = dump_response_input(parsed.output_items)
    # then: URL・パラメータはcompact用になり、返された全項目が再送される
    assert (
        payload.url == "https://example.invalid/v1/responses/compact?api-version=test"
    )
    assert set(payload.body) == {"model", "input", "instructions"}
    assert replay == output
    assert parsed.messages == []


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", "cancelled"])
def test_noncompleted_response_cannot_produce_executable_text(status: str) -> None:
    # テストしたいふるまい: 本文があっても未完了・失敗応答を成功として扱わない。
    # give: 完了していない応答
    response = {"status": status, "output": _output()}
    # when / then: 本文を取り出す前に拒否する
    with pytest.raises(ValueError, match="did not complete"):
        parse_response(response, "responses")


@pytest.mark.parametrize(
    "output",
    [
        [],
        [{"type": "message", "role": "assistant", "content": "invalid"}],
        [
            {
                "type": "message",
                "role": "assistant",
                "status": "incomplete",
                "content": [],
            }
        ],
    ],
)
def test_invalid_output_is_rejected(output: list[dict[str, Any]]) -> None:
    # テストしたいふるまい: 不正な応答構造は本文として実行されない。
    # give: 不正なoutput
    # when / then: 純粋パーサーがエラーを返す
    with pytest.raises(ValueError):
        parse_response({"status": "completed", "output": output}, "responses")


def test_reasoning_is_optional_and_refusal_is_not_executable_text() -> None:
    # テストしたいふるまい: 推論なしの本文を扱い、refusalや推論要約をコード抽出対象にしない。
    # give: 推論なしの通常応答と、拒否のみの応答
    output = _output()[1:]
    refusal = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "cannot comply"}],
        }
    ]
    # when: それぞれの応答を解釈する
    parsed = parse_response({"status": "completed", "output": output}, "responses")
    refused = parse_response({"status": "completed", "output": refusal}, "responses")
    # then: 通常の本文は取得でき、拒否は再送項目にだけ残る
    assert convert_messages_str(parsed.messages) == "hello world!"
    assert refused.messages == []
    assert dump_response_input(refused.output_items) == refusal


def test_chat_completions_remains_the_default_with_legacy_image_format() -> None:
    # テストしたいふるまい: 既存の設定と画像入力形式を引き続き利用できる。
    # give: api_typeを省略した設定と画像メッセージ
    context = create_request_context(
        "https://example.invalid/v1/chat/completions", "test"
    )
    message = MessageContent(
        role="user",
        name="reader",
        content=[
            MessageContentPart(
                type="image_url",
                image_url=ImageURL(url="https://example.invalid/image.png"),
            )
        ],
    )
    # when: リクエストと応答を変換する
    payload = build_request_payload(context, [message])
    parsed = parse_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        context.api_type,
    )
    # then: messages形式と従来の本文が保持され、Responses項目は混入しない
    assert payload.body == {
        "model": "test",
        "messages": [
            {
                "role": "user",
                "name": "reader",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.invalid/image.png",
                            "detail": "auto",
                        },
                    }
                ],
            }
        ],
    }
    assert parsed.messages == [MessageContent(role="assistant", content="ok")]
    assert parsed.output_items == []


@pytest.mark.parametrize(
    "key", ["previous_response_id", "conversation", "input", "messages"]
)
def test_external_history_overrides_are_rejected(key: str) -> None:
    # テストしたいふるまい: ローカル履歴を別の履歴指定で上書きできない。
    # give: 履歴管理と競合するパラメータ
    context = create_request_context(
        "https://example.invalid/v1/responses",
        "test",
        api_type="responses",
        request_params={key: "override"},
    )
    # when / then: 送信前に拒否する
    with pytest.raises(ValueError, match="locally managed"):
        build_request_payload(context, [])


def test_responses_and_compaction_usage_maps_to_existing_model_totals() -> None:
    # テストしたいふるまい: Responsesの入力・出力トークンを推論分の二重加算なく集計する。
    # give: 出力トークンに推論トークンが含まれるusage
    result = APIRequestResult(
        response_json={
            "usage": {
                "input_tokens": 10,
                "output_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 12},
                "total_tokens": 25,
            }
        },
        messages=[],
        resolved_model_name="test",
    )
    # when: 既存の集計形式に変換する
    usage = get_detailed_token_usage_from_response(result)
    # then: 入力10・出力15・合計25として扱う
    assert usage.total_tokens == 25
    assert usage.model_token_usages[0].prompt_tokens == 10
    assert usage.model_token_usages[0].completion_tokens == 15
