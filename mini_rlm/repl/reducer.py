import ast

from mini_rlm.repl.data_model import (
    ReplCommand,
    ReplCommandResult,
    ReplCommandResultType,
    ReplCommandType,
    ReplExecutionState,
    ReplExecutionStatus,
)


def _with_command(
    state: ReplExecutionState,
    command_type: ReplCommandType,
    code: str | None = None,
) -> tuple[ReplExecutionState, ReplCommand]:
    next_state = state.model_copy(update={"last_command_type": command_type})
    return next_state, ReplCommand(type=command_type, code=code)


def _complete(
    state: ReplExecutionState,
) -> tuple[ReplExecutionState, ReplCommand]:
    next_state = state.model_copy(
        update={
            "status": ReplExecutionStatus.COMPLETED,
            "last_command_type": ReplCommandType.COMPLETE,
        }
    )
    return next_state, ReplCommand(type=ReplCommandType.COMPLETE)


def _fail(
    state: ReplExecutionState,
    stderr: str,
) -> tuple[ReplExecutionState, ReplCommand]:
    next_state = state.model_copy(
        update={
            "status": ReplExecutionStatus.FAILED,
            "stderr": stderr,
            "last_command_type": ReplCommandType.EXIT,
        }
    )
    return next_state, ReplCommand(type=ReplCommandType.EXIT)


def _apply_result(
    state: ReplExecutionState,
    result: ReplCommandResult,
) -> ReplExecutionState:
    return state.model_copy(
        update={
            "stdout": state.stdout + result.stdout,
            "stderr": state.stderr + result.stderr,
            "expression_result": result.expression_result
            if result.expression_result is not None
            else state.expression_result,
        }
    )


def _format_syntax_error(error: SyntaxError) -> str:
    message = error.msg
    if error.text is not None:
        message = f"{message} ({error.text.strip()})"
    return f"SyntaxError: {message}"


def _segment_from_lines(
    source_lines: list[str],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> str:
    if start_line == end_line:
        return source_lines[start_line - 1][start_col:end_col]

    segments = [source_lines[start_line - 1][start_col:]]
    for line_number in range(start_line, end_line - 1):
        segments.append(source_lines[line_number])
    segments.append(source_lines[end_line - 1][:end_col])
    return "\n".join(segments)


def _extract_node_source(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment

    source_lines = source.splitlines()
    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    start_col = getattr(node, "col_offset", None)
    end_col = getattr(node, "end_col_offset", None)
    if None in (start_line, end_line, start_col, end_col):
        return ast.unparse(node)
    assert start_line is not None
    assert end_line is not None
    assert start_col is not None
    assert end_col is not None
    return _segment_from_lines(source_lines, start_line, start_col, end_line, end_col)


def _extract_statement_source(source: str, statements: list[ast.stmt]) -> str:
    if not statements:
        return ""

    first_statement = statements[0]
    last_statement = statements[-1]
    source_lines = source.splitlines()
    start_line = getattr(first_statement, "lineno", None)
    end_line = getattr(last_statement, "end_lineno", None)
    start_col = getattr(first_statement, "col_offset", None)
    end_col = getattr(last_statement, "end_col_offset", None)
    if None in (start_line, end_line, start_col, end_col):
        return "\n".join(ast.unparse(statement) for statement in statements)
    assert start_line is not None
    assert end_line is not None
    assert start_col is not None
    assert end_col is not None
    return _segment_from_lines(source_lines, start_line, start_col, end_line, end_col)


def _split_code(
    code: str,
) -> tuple[str, str | None, str | None]:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as error:
        return "", None, _format_syntax_error(error)

    if not module.body:
        return "", None, None

    last_statement = module.body[-1]
    if isinstance(last_statement, ast.Expr):
        statement_code = _extract_statement_source(code, module.body[:-1])
        expression_code = _extract_node_source(code, last_statement.value)
        return statement_code, expression_code, None

    return code, None, None


def reduce_repl_execution(
    prev_state: ReplExecutionState,
    prev_command_result: ReplCommandResult | None,
) -> tuple[ReplExecutionState, ReplCommand]:
    state = prev_state

    if prev_command_result is None:
        statement_code, final_expression_code, syntax_error = _split_code(state.code)
        if syntax_error is not None:
            return _fail(state, syntax_error)

        state = state.model_copy(
            update={
                "statement_code": statement_code,
                "final_expression_code": final_expression_code,
            }
        )

        if statement_code != "":
            return _with_command(
                state,
                ReplCommandType.EXECUTE_STATEMENTS,
                statement_code,
            )

        if final_expression_code is not None:
            return _with_command(
                state,
                ReplCommandType.EVALUATE_EXPRESSION,
                final_expression_code,
            )

        return _complete(state)

    state = _apply_result(state, prev_command_result)

    if prev_command_result.type != ReplCommandResultType.SUCCESS:
        return _fail(state, state.stderr)

    if prev_command_result.command_type == ReplCommandType.EXECUTE_STATEMENTS:
        if state.final_expression_code is not None:
            return _with_command(
                state,
                ReplCommandType.EVALUATE_EXPRESSION,
                state.final_expression_code,
            )
        return _complete(state)

    if prev_command_result.command_type == ReplCommandType.EVALUATE_EXPRESSION:
        return _complete(state)

    return _with_command(state, ReplCommandType.EXIT)
