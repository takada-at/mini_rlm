from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommandType,
    ReplSessionLimits,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
    TerminationReason,
)
from mini_rlm.repl_session.reducer import reduce_repl_session


def build_state(**updates: object) -> ReplSessionState:
    base = ReplSessionState(
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


def test_reduce_repl_session_history_over_limit_returns_compacting() -> None:
    # give: 履歴長が上限を超えている
    prev_state = build_state(history_length=6)
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
