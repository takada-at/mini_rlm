from mini_rlm.llm import (
    HistoryItem,
    MessageContent,
    ResponseItem,
    create_request_context,
)
from mini_rlm.llm.protocol import build_request_payload, dump_response_input
from mini_rlm.repl import ReplResult
from mini_rlm.repl_session.convert import build_session_messages
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommandType,
    ReplSessionHistoryEntry,
    ReplSessionLimits,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
)
from mini_rlm.repl_session.executor_command import format_iteration
from mini_rlm.repl_session.reducer import reduce_repl_session


def _state() -> ReplSessionState:
    return ReplSessionState(
        prompt="inspect the file",
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=1000, iteration_limit=10, timeout_seconds=60, error_threshold=3
        ),
        started_at_seconds=0,
        current_time_seconds=1,
    )


def _items() -> list[HistoryItem]:
    return [
        ResponseItem.model_validate(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "opaque",
            }
        ),
        ResponseItem.model_validate(
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "```python\nprint(1)\n```",
                        "annotations": [],
                    }
                ],
            }
        ),
    ]


def test_next_iteration_replays_reasoning_before_execution_results_once() -> None:
    # テストしたいふるまい: API応答からreducerを経て次の入力まで、推論と実行結果を順序どおり一度だけ保持する。
    # give: 推論を含む応答とコード実行結果
    state = _state()
    items = _items()
    llm_result = CommandResult(
        command_type=ReplSessionCommandType.CALL_LLM,
        type=ReplSessionResultType.SUCCESS,
        last_llm_message="```python\nprint(1)\n```",
        last_llm_items=items,
    )
    execution = ReplSessionHistoryEntry(
        code="print(1)",
        repl_result=ReplResult(stdout="1", stderr="", execution_time=0, locals={}),
    )
    # when: CALL_LLM -> EXECUTE_CODE -> APPEND_HISTORYを処理して次の入力を作る
    called, _ = reduce_repl_session(state, llm_result)
    executed, _ = reduce_repl_session(
        called,
        CommandResult(
            command_type=ReplSessionCommandType.EXECUTE_CODE,
            type=ReplSessionResultType.SUCCESS,
            repl_results=[execution],
        ),
    )
    formatted = format_iteration(
        executed.last_llm_message or "",
        executed.repl_results or [],
        response_items=executed.last_llm_items,
    )
    appended, _ = reduce_repl_session(
        executed,
        CommandResult(
            command_type=ReplSessionCommandType.APPEND_HISTORY,
            type=ReplSessionResultType.SUCCESS,
            new_messages=formatted,
        ),
    )
    context = create_request_context(
        "https://example.invalid/v1/responses", "test", api_type="responses"
    )
    payload = build_request_payload(
        context,
        build_session_messages(appended, context, "system"),
        include_context_messages=False,
    )
    # then: system・質問・推論・応答・実行結果の順になり、元の状態を変更しない
    assert payload.body["input"][2:4] == dump_response_input(items)
    assert len(payload.body["input"]) == 5
    assert (
        payload.body["input"][4]["content"][0]["text"].split("REPL output:")[1].strip()
        == "1"
    )
    assert state.messages is None
    assert called.messages is None
    assert len(items) == 2


def test_compaction_replaces_history_without_repeating_initial_messages() -> None:
    # テストしたいふるまい: 圧縮結果全体を採用し、初期質問・contextのユーザー入力を重複送信しない。
    # give: 初期prefix、古い履歴、ユーザー入力を保持した圧縮結果
    context = create_request_context(
        "https://example.invalid/v1/responses", "test", api_type="responses"
    )
    context.messages = [
        MessageContent(role="system", content="prefix instruction"),
        MessageContent(role="user", content="prefix data"),
    ]
    state = _state().model_copy(
        update={"messages": _items(), "current_history_tokens": 900}
    )
    compacted: list[HistoryItem] = [
        ResponseItem.model_validate(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "prefix data"}],
            }
        ),
        ResponseItem.model_validate(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": state.prompt}],
            }
        ),
        ResponseItem.model_validate(
            {"type": "compaction", "id": "cmp_1", "encrypted_content": "compressed"}
        ),
    ]
    # when: 圧縮成功を適用して次のリクエストを組み立てる
    next_state, command = reduce_repl_session(
        state,
        CommandResult(
            command_type=ReplSessionCommandType.COMPACTING,
            type=ReplSessionResultType.SUCCESS,
            compacted_messages=compacted,
            history_includes_prompt=True,
            consumed_tokens=20,
        ),
    )
    messages = build_session_messages(next_state, context, "rlm instruction")
    payload = build_request_payload(context, messages, include_context_messages=False)
    # then: システム指示と圧縮結果だけを送り、古い推論や初期質問を追加しない
    assert command.type == ReplSessionCommandType.CALL_LLM
    assert payload.body["input"][2:] == dump_response_input(compacted)
    assert len(payload.body["input"]) == 5
    assert [item["role"] for item in payload.body["input"][:2]] == ["system", "system"]
    assert next_state.current_history_tokens == 0
    assert next_state.total_tokens == 20
    assert state.messages == _items()


def test_compaction_failure_keeps_history_and_retries_same_command() -> None:
    # テストしたいふるまい: 圧縮失敗で推論履歴を失わず、同じ履歴で再試行する。
    # give: 圧縮対象の履歴
    state = _state().model_copy(
        update={"messages": _items(), "current_history_tokens": 900}
    )
    # when: 圧縮エラーを適用する
    next_state, command = reduce_repl_session(
        state,
        CommandResult(
            command_type=ReplSessionCommandType.COMPACTING,
            type=ReplSessionResultType.ERROR,
            error_message="unavailable",
        ),
    )
    # then: 履歴とprefix管理を保持し、エラーを数えて圧縮を再試行する
    assert command.type == ReplSessionCommandType.COMPACTING
    assert next_state.messages == state.messages
    assert not next_state.history_includes_prompt
    assert next_state.error_count == 1


def test_failed_llm_call_does_not_append_items_or_advance_to_code_execution() -> None:
    # テストしたいふるまい: 不完全な応答による失敗で履歴を汚さず、コード実行に進まない。
    # give: 前反復の履歴
    state = _state().model_copy(update={"messages": _items()})
    # when: CALL_LLMが失敗する
    next_state, command = reduce_repl_session(
        state,
        CommandResult(
            command_type=ReplSessionCommandType.CALL_LLM,
            type=ReplSessionResultType.ERROR,
            error_message="incomplete",
        ),
    )
    # then: 同じ履歴からCALL_LLMをやり直す
    assert command.type == ReplSessionCommandType.CALL_LLM
    assert next_state.messages == state.messages
    assert next_state.last_llm_items == []
