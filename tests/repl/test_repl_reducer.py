from mini_rlm.repl.data_model import (
    ReplCommandResult,
    ReplCommandResultType,
    ReplCommandType,
    ReplExecutionState,
    ReplExecutionStatus,
)
from mini_rlm.repl.reducer import reduce_repl_execution


def _state(code: str) -> ReplExecutionState:
    return ReplExecutionState(code=code, status=ReplExecutionStatus.RUNNING)


def _success_result(
    command_type: ReplCommandType,
    *,
    stdout: str = "",
    stderr: str = "",
    expression_result: str | None = None,
) -> ReplCommandResult:
    return ReplCommandResult(
        command_type=command_type,
        type=ReplCommandResultType.SUCCESS,
        stdout=stdout,
        stderr=stderr,
        expression_result=expression_result,
    )


def _error_result(
    command_type: ReplCommandType,
    *,
    stdout: str = "",
    stderr: str = "",
) -> ReplCommandResult:
    return ReplCommandResult(
        command_type=command_type,
        type=ReplCommandResultType.ERROR,
        stdout=stdout,
        stderr=stderr,
    )


def test_reduce_repl_execution_starts_with_statements_when_last_line_is_expression() -> (
    None
):
    # given: 末尾が式のコードがある
    prev_state = _state("x = 1\nx + 1")
    # when: 初回の reducer を呼ぶ
    next_state, command = reduce_repl_execution(prev_state, None)
    # then: 末尾式を分離して先に文を実行する
    assert next_state.statement_code == "x = 1"
    assert next_state.final_expression_code == "x + 1"
    assert command.type == ReplCommandType.EXECUTE_STATEMENTS
    assert command.code == "x = 1"


def test_reduce_repl_execution_starts_with_eval_when_code_is_single_expression() -> (
    None
):
    # given: コード全体が単一の式である
    prev_state = _state("1 + 1")
    # when: 初回の reducer を呼ぶ
    next_state, command = reduce_repl_execution(prev_state, None)
    # then: 文実行を挟まず式評価を指示する
    assert next_state.statement_code == ""
    assert next_state.final_expression_code == "1 + 1"
    assert command.type == ReplCommandType.EVALUATE_EXPRESSION
    assert command.code == "1 + 1"


def test_reduce_repl_execution_advances_to_eval_after_statement_execution() -> None:
    # given: 文実行後に式評価が必要な状態である
    parsed_state, _ = reduce_repl_execution(_state("x = 1\nx + 1"), None)
    prev_result = _success_result(ReplCommandType.EXECUTE_STATEMENTS, stdout="ok\n")
    # when: 文実行成功を reducer に渡す
    next_state, command = reduce_repl_execution(parsed_state, prev_result)
    # then: 次は末尾式の評価に進む
    assert next_state.stdout == "ok\n"
    assert command.type == ReplCommandType.EVALUATE_EXPRESSION
    assert command.code == "x + 1"


def test_reduce_repl_execution_completes_after_expression_evaluation() -> None:
    # given: 末尾式の評価直前まで進んでいる
    parsed_state, _ = reduce_repl_execution(_state("x = 1\nx + 1"), None)
    eval_state, _ = reduce_repl_execution(
        parsed_state,
        _success_result(ReplCommandType.EXECUTE_STATEMENTS),
    )
    prev_result = _success_result(
        ReplCommandType.EVALUATE_EXPRESSION,
        expression_result="2",
    )
    # when: 式評価成功を reducer に渡す
    next_state, command = reduce_repl_execution(eval_state, prev_result)
    # then: 評価結果を保持して完了する
    assert next_state.status == ReplExecutionStatus.COMPLETED
    assert next_state.expression_result == "2"
    assert command.type == ReplCommandType.COMPLETE


def test_reduce_repl_execution_completes_after_statement_only_code() -> None:
    # given: 末尾式を持たないコードである
    parsed_state, command = reduce_repl_execution(_state("x = 1"), None)
    # when: 文実行成功を reducer に渡す
    next_state, complete_command = reduce_repl_execution(
        parsed_state,
        _success_result(ReplCommandType.EXECUTE_STATEMENTS),
    )
    # then: 追加評価なしで完了する
    assert command.type == ReplCommandType.EXECUTE_STATEMENTS
    assert next_state.status == ReplExecutionStatus.COMPLETED
    assert complete_command.type == ReplCommandType.COMPLETE


def test_reduce_repl_execution_fails_fast_on_syntax_error() -> None:
    # given: AST パースできないコードである
    prev_state = _state("x =")
    # when: 初回の reducer を呼ぶ
    next_state, command = reduce_repl_execution(prev_state, None)
    # then: 構文エラーとして終了する
    assert next_state.status == ReplExecutionStatus.FAILED
    assert "SyntaxError" in next_state.stderr
    assert command.type == ReplCommandType.EXIT


def test_reduce_repl_execution_exits_on_command_error() -> None:
    # given: 文実行フェーズにいる
    parsed_state, _ = reduce_repl_execution(_state("x = 1\nx + 1"), None)
    prev_result = _error_result(
        ReplCommandType.EXECUTE_STATEMENTS,
        stdout="partial\n",
        stderr="boom",
    )
    # when: エラー結果を reducer に渡す
    next_state, command = reduce_repl_execution(parsed_state, prev_result)
    # then: 失敗状態で終了する
    assert next_state.status == ReplExecutionStatus.FAILED
    assert next_state.stdout == "partial\n"
    assert next_state.stderr == "boom"
    assert command.type == ReplCommandType.EXIT
