import contextlib
import io
import os
import sys
from time import perf_counter
from typing import Any

from mini_rlm.repl.data_model import (
    ReplCommand,
    ReplCommandResult,
    ReplCommandResultType,
    ReplCommandType,
    ReplExecutionState,
    ReplExecutionStatus,
    ReplResult,
    ReplState,
)
from mini_rlm.repl.reducer import reduce_repl_execution


@contextlib.contextmanager
def _capture_output(state: ReplState):
    with state.lock:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            yield stdout_buf, stderr_buf
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


@contextlib.contextmanager
def _temp_cwd(state: ReplState):
    old_cwd = os.getcwd()
    try:
        os.chdir(state.temp_dir)
        yield
    finally:
        os.chdir(old_cwd)


def _restore_scaffold(state: ReplState) -> None:
    for name, value in state.reserved_globals.items():
        state.globals[name] = value

    if "context_0" in state.locals and "context" not in state.reserved_globals:
        state.locals["context"] = state.locals["context_0"]
    if "history_0" in state.locals and "history" not in state.reserved_globals:
        state.locals["history"] = state.locals["history_0"]


def _combine_namespace(state: ReplState) -> dict[str, Any]:
    return {**state.globals, **state.locals}


def _promote_user_defined_names(
    state: ReplState,
    combined_namespace: dict[str, Any],
) -> None:
    for key, value in combined_namespace.items():
        if key not in state.globals and not key.startswith("_"):
            state.locals[key] = value


def _result_from_command(
    command: ReplCommand,
    *,
    stdout: str = "",
    stderr: str = "",
    expression_result: str | None = None,
    result_type: ReplCommandResultType = ReplCommandResultType.SUCCESS,
) -> ReplCommandResult:
    return ReplCommandResult(
        command_type=command.type,
        type=result_type,
        stdout=stdout,
        stderr=stderr,
        expression_result=expression_result,
    )


def _execute_statements(
    repl_state: ReplState,
    command: ReplCommand,
) -> ReplCommandResult:
    code = command.code or ""
    with _capture_output(repl_state) as (stdout_buf, stderr_buf), _temp_cwd(repl_state):
        combined = _combine_namespace(repl_state)
        try:
            exec(code, combined, combined)  # noqa: S102
            _promote_user_defined_names(repl_state, combined)
            _restore_scaffold(repl_state)
            return _result_from_command(
                command,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
            )
        except Exception as error:
            _promote_user_defined_names(repl_state, combined)
            _restore_scaffold(repl_state)
            return _result_from_command(
                command,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue() + f"\n{type(error).__name__}: {error}",
                result_type=ReplCommandResultType.ERROR,
            )


def _display_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _evaluate_expression(
    repl_state: ReplState,
    command: ReplCommand,
) -> ReplCommandResult:
    code = command.code or ""
    with _capture_output(repl_state) as (stdout_buf, stderr_buf), _temp_cwd(repl_state):
        combined = _combine_namespace(repl_state)
        try:
            value = eval(code, combined, combined)  # noqa: S307
            _promote_user_defined_names(repl_state, combined)
            _restore_scaffold(repl_state)
            return _result_from_command(
                command,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                expression_result=_display_value(value),
            )
        except Exception as error:
            _promote_user_defined_names(repl_state, combined)
            _restore_scaffold(repl_state)
            return _result_from_command(
                command,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue() + f"\n{type(error).__name__}: {error}",
                result_type=ReplCommandResultType.ERROR,
            )


def _execute_command(
    repl_state: ReplState,
    command: ReplCommand,
) -> ReplCommandResult:
    if command.type == ReplCommandType.EXECUTE_STATEMENTS:
        return _execute_statements(repl_state, command)
    if command.type == ReplCommandType.EVALUATE_EXPRESSION:
        return _evaluate_expression(repl_state, command)
    return _result_from_command(
        command,
        stderr="Unknown command",
        result_type=ReplCommandResultType.ERROR,
    )


def execute_repl_execution(
    repl_state: ReplState,
    code: str,
) -> ReplResult:
    start_time = perf_counter()
    execution_state = ReplExecutionState(
        code=code,
        status=ReplExecutionStatus.RUNNING,
    )
    prev_result: ReplCommandResult | None = None

    while True:
        execution_state, command = reduce_repl_execution(execution_state, prev_result)

        if command.type in (ReplCommandType.COMPLETE, ReplCommandType.EXIT):
            final_answer = repl_state.last_final_answer
            repl_state.last_final_answer = None
            return ReplResult(
                stdout=execution_state.stdout,
                stderr=execution_state.stderr,
                locals=repl_state.locals.copy(),
                execution_time=perf_counter() - start_time,
                final_answer=final_answer,
                expression_result=execution_state.expression_result,
            )

        prev_result = _execute_command(repl_state, command)
