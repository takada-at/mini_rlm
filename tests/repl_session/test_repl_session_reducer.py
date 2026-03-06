from mini_rlm.llm.data_model import MessageContent
from mini_rlm.repl.data_model import ReplResult
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommandType,
    ReplSessionHistoryEntry,
    ReplSessionLimits,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
    TerminationReason,
)
from mini_rlm.repl_session.reducer import reduce_repl_session


def build_state(**updates: object) -> ReplSessionState:
    base = ReplSessionState(
        prompt="test prompt",
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=100,
            iteration_limit=3,
            timeout_seconds=60.0,
            error_threshold=2,
            history_limit=5,
        ),
        started_at_seconds=0.0,
        current_time_seconds=10.0,
    )
    return base.model_copy(update=updates)


def test_reduce_repl_session_initial_step_returns_call_llm() -> None:
    # give: 初期stateで履歴上限を超えていない
    prev_state = build_state()
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: call_llmコマンドが返る
    assert command.type == ReplSessionCommandType.CALL_LLM
    assert next_state.last_command_type == ReplSessionCommandType.CALL_LLM


def test_reduce_repl_session_success_path_advances_commands() -> None:
    # give: call_llm成功後の結果
    prev_state = build_state(last_command_type=ReplSessionCommandType.CALL_LLM)
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.CALL_LLM,
        type=ReplSessionResultType.SUCCESS,
        consumed_tokens=10,
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, prev_result)
    # then: execute_codeへ遷移する
    assert command.type == ReplSessionCommandType.EXECUTE_CODE
    assert next_state.total_tokens == 10


def test_call_llm_update_last_llm_message() -> None:
    # give: call_llm成功後の結果
    prev_state = build_state(last_command_type=ReplSessionCommandType.CALL_LLM)
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.CALL_LLM,
        type=ReplSessionResultType.SUCCESS,
        consumed_tokens=10,
        last_llm_message="Hello, world!",
    )
    # when: reducerを実行する
    next_state, _ = reduce_repl_session(prev_state, prev_result)
    # then: last_llm_messageが更新される
    assert next_state.last_llm_message == "Hello, world!"


def test_execute_code_update_repl_result() -> None:
    # give: execute_code成功後の結果
    prev_state = build_state(last_command_type=ReplSessionCommandType.EXECUTE_CODE)
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.EXECUTE_CODE,
        type=ReplSessionResultType.SUCCESS,
        repl_results=[
            ReplSessionHistoryEntry(
                code="print('output')",
                repl_result=ReplResult(
                    stdout="output", locals={"x": 1}, stderr="", execution_time=0.1
                ),
            )
        ],
    )
    # when: reducerを実行する
    next_state, _ = reduce_repl_session(prev_state, prev_result)
    # then: repl_resultが更新される
    assert next_state.repl_results is not None
    assert len(next_state.repl_results) == 1
    assert next_state.repl_results[0] is not None
    assert next_state.repl_results[0].repl_result is not None
    assert next_state.repl_results[0].repl_result.stdout == "output"
    assert next_state.repl_results[0].repl_result.locals == {"x": 1}


def test_append_history_update_messages() -> None:
    # give: append_history成功後の結果
    prev_state = build_state(
        last_command_type=ReplSessionCommandType.APPEND_HISTORY,
        messages=[
            MessageContent(role="user", content="start"),
        ],
    )
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.APPEND_HISTORY,
        type=ReplSessionResultType.SUCCESS,
        consumed_tokens=0,
        new_messages=[
            MessageContent(
                role="user",
                content="print('hello')",
            ),
            MessageContent(
                role="user",
                content="print('world')",
            ),
        ],
    )
    # when: reducerを実行する
    next_state, _ = reduce_repl_session(prev_state, prev_result)
    # then: messagesが更新される
    assert next_state.messages is not None
    assert len(next_state.messages) == 3
    assert next_state.messages[0].content == "start"
    assert next_state.messages[1].content == "print('hello')"
    assert next_state.messages[2].content == "print('world')"


def test_reduce_repl_session_history_over_limit_returns_compacting() -> None:
    # give: 履歴長が上限を超えている
    prev_state = build_state(
        messages=[
            MessageContent(role="user", content=f"message {i}") for i in range(11)
        ],
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: compactingコマンドが返る
    assert command.type == ReplSessionCommandType.COMPACTING
    assert next_state.last_command_type == ReplSessionCommandType.COMPACTING


def test_reduce_repl_session_check_complete_success_and_not_complete_starts_next_iteration() -> (
    None
):
    # give: check_complete成功で未完了
    prev_state = build_state(
        iteration_count=1,
        last_command_type=ReplSessionCommandType.CHECK_COMPLETE,
    )
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.CHECK_COMPLETE,
        type=ReplSessionResultType.SUCCESS,
        is_complete=False,
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, prev_result)
    # then: iterationが進みcall_llmに戻る
    assert next_state.iteration_count == 2
    assert command.type == ReplSessionCommandType.CALL_LLM


def test_reduce_repl_session_check_complete_success_and_complete_returns_complete_command() -> (
    None
):
    # give: check_complete成功で完了判定
    prev_state = build_state(last_command_type=ReplSessionCommandType.CHECK_COMPLETE)
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.CHECK_COMPLETE,
        type=ReplSessionResultType.SUCCESS,
        final_answer="final answer",
        is_complete=True,
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, prev_result)
    # then: completeコマンドで終了する
    assert command.type == ReplSessionCommandType.COMPLETE
    assert next_state.status == ReplSessionStatus.COMPLETED
    assert next_state.termination_reason == TerminationReason.COMPLETED


def test_reduce_repl_session_token_limit_exceeded_returns_exit() -> None:
    # give: token上限を超えている
    prev_state = build_state(total_tokens=101)
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: token上限超過でexitする
    assert command.type == ReplSessionCommandType.EXIT
    assert next_state.status == ReplSessionStatus.FAILED
    assert next_state.termination_reason == TerminationReason.TOKEN_LIMIT_EXCEEDED


def test_reduce_repl_session_iterations_exhausted_returns_exit() -> None:
    # give: iteration上限に達している
    prev_state = build_state(iteration_count=3)
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: iteration上限超過でexitする
    assert command.type == ReplSessionCommandType.EXIT
    assert next_state.termination_reason == TerminationReason.ITERATIONS_EXHAUSTED


def test_reduce_repl_session_timeout_returns_exit() -> None:
    # give: timeout秒数を超過している
    prev_state = build_state(current_time_seconds=61.0)
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: timeoutでexitする
    assert command.type == ReplSessionCommandType.EXIT
    assert next_state.termination_reason == TerminationReason.TIMEOUT


def test_reduce_repl_session_error_threshold_exceeded_returns_exit() -> None:
    # give: 直前コマンドが失敗し、閾値到達する
    prev_state = build_state(
        error_count=1,
        last_command_type=ReplSessionCommandType.CALL_LLM,
    )
    prev_result = CommandResult(
        command_type=ReplSessionCommandType.CALL_LLM,
        type=ReplSessionResultType.ERROR,
        error_message="llm error",
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, prev_result)
    # then: error_threshold超過でexitする
    assert command.type == ReplSessionCommandType.EXIT
    assert next_state.error_count == 2
    assert next_state.termination_reason == TerminationReason.ERROR_THRESHOLD_EXCEEDED


def test_reduce_repl_session_cancelled_returns_exit() -> None:
    # give: キャンセル済みフラグが立っている
    prev_state = build_state(is_cancelled=True)
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, None)
    # then: cancelledでexitする
    assert command.type == ReplSessionCommandType.EXIT
    assert next_state.termination_reason == TerminationReason.CANCELLED


def test_compaction_command_truncates_messages() -> None:
    # give: compactingコマンド実行時に履歴長が上限を超えている
    prev_state = build_state(
        history_length=11, last_command_type=ReplSessionCommandType.COMPACTING
    )
    command_result = CommandResult(
        command_type=ReplSessionCommandType.COMPACTING,
        type=ReplSessionResultType.SUCCESS,
        compacted_messages=[
            MessageContent(role="user", content=f"message {i}") for i in range(5)
        ],
    )
    # when: reducerを実行する
    next_state, command = reduce_repl_session(prev_state, command_result)
    # then: messagesが履歴上限の半分にトランケートされる
    assert next_state.messages is not None
    assert len(next_state.messages) == 5
