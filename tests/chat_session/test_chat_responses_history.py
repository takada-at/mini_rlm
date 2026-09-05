import pytest

from mini_rlm.chat_session.convert import build_decision_messages
from mini_rlm.chat_session.data_model import (
    ChatDecision,
    ChatDecisionType,
    ChatSessionCommandType,
    ChatSessionResultType,
    ChatSessionState,
    CommandResult,
)
from mini_rlm.chat_session.executor import reset_chat_session
from mini_rlm.chat_session.reducer import reduce_chat_session
from mini_rlm.llm import (
    HistoryItem,
    MessageContent,
    ResponseItem,
    create_request_context,
)
from mini_rlm.llm.protocol import dump_response_input


def _state() -> ChatSessionState:
    context = create_request_context(
        "https://example.invalid/v1/responses", "test", api_type="responses"
    )
    return ChatSessionState(
        chat_request_context=context,
        run_request_context=context,
        pending_user_text="hello",
    )


def _decision_result(state: ChatSessionState, *, run: bool = False) -> CommandResult:
    text = (
        '{"type":"run_agent","task":"inspect"}'
        if run
        else '{"type":"respond_chat","message":"Hi"}'
    )
    new_history: list[HistoryItem] = [
        build_decision_messages(state)[-1],
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
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ),
    ]
    return CommandResult(
        command_type=ChatSessionCommandType.DECIDE,
        type=ChatSessionResultType.SUCCESS,
        new_history=new_history,
        decision=ChatDecision(
            type=ChatDecisionType.RUN_AGENT if run else ChatDecisionType.RESPOND_CHAT,
            message=None if run else "Hi",
            task="inspect" if run else None,
        ),
    )


def test_next_chat_turn_replays_original_input_and_reasoning_with_decision_json() -> (
    None
):
    # テストしたいふるまい: 次ターンで表示用の回答へ置き換えず、実際の入力と推論・判断JSONを送る。
    # give: 推論付き判断応答
    initial = _state()
    result = _decision_result(initial)
    # when: ターンを確定して次の入力を作る
    finished, _ = reduce_chat_session(initial, result)
    next_turn = finished.model_copy(update={"pending_user_text": "continue"})
    messages = build_decision_messages(next_turn)
    # then: 元のユーザー入力・推論・判断JSONを順序どおり再送し、表示本文の重複はない
    assert dump_response_input(messages[1:-1]) == dump_response_input(
        result.new_history
    )
    assert len(messages) == 5
    assert finished.turns[-1].assistant_text == "Hi"
    assert initial.api_history is None
    assert initial.chat_request_context.messages is None


@pytest.mark.parametrize("failed", [False, True])
def test_agent_result_is_recorded_after_decision_items(failed: bool) -> None:
    # テストしたいふるまい: エージェント実行の成功・失敗を判断の推論に続けて次ターンへ渡す。
    # give: RUN_AGENTを選んだ判断
    initial = _state()
    decision = _decision_result(initial, run=True)
    decided, _ = reduce_chat_session(initial, decision)
    result = CommandResult(
        command_type=ChatSessionCommandType.RUN_AGENT,
        type=ChatSessionResultType.ERROR if failed else ChatSessionResultType.SUCCESS,
        assistant_text="answer",
        error_message="execution failed" if failed else None,
    )
    # when: 実行結果を適用する
    finished, _ = reduce_chat_session(decided, result)
    # then: 元の入力を重複させず、実際の実行結果を追加する
    assert finished.api_history is not None
    assert finished.api_history[:-1] == decision.new_history
    outcome = finished.api_history[-1]
    assert isinstance(outcome, MessageContent)
    assert outcome.content == "Agent execution result:\n" + (
        "execution failed" if failed else "answer"
    )
    assert len(decided.api_history or []) == 3


@pytest.mark.parametrize("failed", [False, True])
def test_forced_run_records_input_and_result_without_inventing_reasoning(
    failed: bool,
) -> None:
    # テストしたいふるまい: 判断をスキップした/runではユーザー入力と実行結果のみを保持する。
    # give: 強制実行の状態
    state = _state().model_copy(
        update={
            "pending_decision": ChatDecision(
                type=ChatDecisionType.RUN_AGENT, task="inspect"
            )
        }
    )
    # when: 判断を呼ばずに実行して完了する
    _, command = reduce_chat_session(state, None)
    finished, _ = reduce_chat_session(
        state,
        CommandResult(
            command_type=ChatSessionCommandType.RUN_AGENT,
            type=ChatSessionResultType.ERROR
            if failed
            else ChatSessionResultType.SUCCESS,
            assistant_text="answer",
            error_message="failed" if failed else None,
        ),
    )
    # then: 架空の判断応答や推論を追加しない
    assert command.type == ChatSessionCommandType.RUN_AGENT
    assert finished.api_history is not None
    assert len(finished.api_history) == 2
    assert finished.api_history[0] == MessageContent(role="user", content="hello")
    assert all(isinstance(item, MessageContent) for item in finished.api_history)


def test_invalid_decision_preserves_received_items_and_adds_failure_context() -> None:
    # テストしたいふるまい: JSON解釈失敗でも受信済みの推論を保持し、次ターンに失敗を伝える。
    # give: API応答は得られたが判断の解釈に失敗した結果
    state = _state()
    result = _decision_result(state).model_copy(
        update={
            "type": ChatSessionResultType.ERROR,
            "decision": None,
            "error_message": "invalid JSON",
        }
    )
    # when: 失敗ターンを確定する
    finished, _ = reduce_chat_session(state, result)
    # then: 受信済み項目の後に失敗理由を追加し、入力は重複しない
    assert finished.api_history is not None
    assert finished.api_history[:-1] == result.new_history
    assert finished.api_history[-1] == MessageContent(
        role="user", content="Chat turn error:\ninvalid JSON"
    )


def test_reset_clears_reasoning_and_does_not_affect_other_sessions() -> None:
    # テストしたいふるまい: /resetで推論を消去し、共有contextを使う別セッションの履歴は独立する。
    # give: 応答を保持したセッションと同じcontextを使う別セッション
    initial = _state()
    finished, _ = reduce_chat_session(initial, _decision_result(initial))
    other = ChatSessionState(
        chat_request_context=initial.chat_request_context,
        run_request_context=initial.run_request_context,
        pending_user_text="other",
    )
    # when: セッションをリセットする
    reset = reset_chat_session(finished)
    # then: 他のセッションやリセット後の入力に過去の推論は混入しない
    assert reset.api_history is None
    assert reset.turns == []
    assert reset.pending_input_recorded is False
    assert len(build_decision_messages(other)) == 2
    assert len(finished.api_history or []) == 3
    assert initial.chat_request_context.messages is None
