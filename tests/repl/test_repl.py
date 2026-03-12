import pytest

from mini_rlm.repl.repl import (
    add_context,
    add_function,
    add_history,
    cleanup,
    create_repl,
    execute_code,
    load_context,
    show_vars,
)


@pytest.fixture
def state():
    s = create_repl()
    yield s
    cleanup(s)


# =============================================================================
# execute_code
# =============================================================================


def test_execute_code_returns_stdout(state):
    # given: REPL が初期化されている
    # when: print を含むコードを実行する
    result = execute_code(state, "print('hello')")
    # then: stdout に出力が含まれる
    assert result.stdout == "hello\n"
    assert result.stderr == ""


def test_execute_code_persists_variables_across_calls(state):
    # given: 1回目の実行で変数を定義している
    execute_code(state, "x = 42")
    # when: 2回目の実行で同じ変数を参照する
    result = execute_code(state, "print(x)")
    # then: 前回のセッションで定義した値が参照できる
    assert result.stdout == "42\n"


def test_execute_code_captures_exception_in_stderr(state):
    # given: REPL が初期化されている
    # when: 例外が発生するコードを実行する
    result = execute_code(state, "1 / 0")
    # then: stderr にエラーメッセージが含まれ、stdout は空
    assert "ZeroDivisionError" in result.stderr
    assert result.stdout == ""


def test_execute_code_returns_locals_snapshot(state):
    # given: REPL が初期化されている
    # when: 変数を定義するコードを実行する
    result = execute_code(state, "a = 1\nb = 2")
    # then: REPLResult.locals にその時点の変数が含まれる
    assert result.locals["a"] == 1
    assert result.locals["b"] == 2


def test_execute_code_records_execution_time(state):
    # given: REPL が初期化されている
    # when: コードを実行する
    result = execute_code(state, "pass")
    # then: execution_time が非負の数値として記録される
    assert result.execution_time >= 0


def test_execute_code_blocked_builtins(state):
    # given: eval / exec / input がブロックされている
    # when: eval を呼び出そうとする
    result = execute_code(state, "eval('1+1')")
    # then: TypeError が stderr に出る（eval=None なので呼び出し不可）
    assert result.stderr != ""


def test_execute_code_returns_last_expression_result(state):
    # given: 変数を定義して末尾を式で終えるコードである
    # when: コードを実行する
    result = execute_code(state, "x = 40\nx + 2")
    # then: 末尾式の評価結果が返る
    assert result.expression_result == "42"
    assert result.locals["x"] == 40


def test_execute_code_uses_overwritten_global_name_for_last_expression(state):
    # given: REPL グローバルに関数が登録されている
    add_function(state, "double", lambda x: x * 2)
    # when: 同一コードブロック内で同名変数に上書きして末尾式で参照する
    result = execute_code(state, "double = 0\ndouble + 1")
    # then: 末尾式は上書き後の値を参照し、次回実行では関数が復元される
    assert result.expression_result == "1"
    restored = execute_code(state, "print(double(21))")
    assert restored.stdout == "42\n"


def test_execute_code_returns_syntax_error_in_stderr(state):
    # given: 構文エラーを含むコードである
    # when: コードを実行する
    result = execute_code(state, "x =")
    # then: 構文エラーが stderr に返る
    assert "SyntaxError" in result.stderr


def test_execute_code_returns_consumed_tokens_delta_for_helper_calls(state):
    # given: token usage を ledger に積む helper が登録されている
    def consume_tokens() -> str:
        state.usage_ledger.total_consumed_tokens += 17
        return "ok"

    add_function(state, "consume_tokens", consume_tokens)
    # when: REPL 内で helper を呼ぶ
    result = execute_code(state, "value = consume_tokens()\nprint(value)")
    followup = execute_code(state, "print('next')")
    # then: 実行中の token 増分だけが ReplResult.consumed_tokens に入る
    assert result.stdout == "ok\n"
    assert result.consumed_tokens == 17
    assert followup.consumed_tokens == 0


# =============================================================================
# FINAL_VAR
# =============================================================================


def test_final_var_with_existing_variable(state):
    # given: 変数 result が定義されている
    execute_code(state, "result = 'answer'")
    # when: FINAL_VAR でその変数名を指定する
    r = execute_code(state, "FINAL_VAR('result')")
    # then: final_answer にその値が文字列で入る
    assert r.final_answer == "answer"


def test_final_var_with_direct_value(state):
    # given: REPL が初期化されている
    # when: FINAL_VAR に変数名でなく直接値を渡す
    r = execute_code(state, "FINAL_VAR(123)")
    # then: final_answer にその値が文字列化されて入る
    assert r.final_answer == "123"


def test_final_var_with_missing_variable_returns_error_string(state):
    # given: 変数 missing は存在しない
    # when: FINAL_VAR でその変数名を指定する
    r = execute_code(state, "FINAL_VAR('missing')")
    # then: final_answer は None のまま、stdout/stderr にエラーメッセージが出る
    assert r.final_answer is None


def test_final_var_overwrite_is_restored(state):
    # given: ユーザーコードが FINAL_VAR を上書きしている
    execute_code(state, "FINAL_VAR = None")
    # when: 次の実行で FINAL_VAR を使う
    execute_code(state, "x = 'ok'")
    r = execute_code(state, "FINAL_VAR('x')")
    # then: scaffold がリストアされているので正常に動作する
    assert r.final_answer == "ok"


# =============================================================================
# SHOW_VARS
# =============================================================================


def test_show_vars_lists_defined_variables(state):
    # given: いくつかの変数が定義されている
    execute_code(state, "foo = 1\nbar = 'baz'")
    # when: SHOW_VARS を呼ぶ
    output = show_vars(state)
    # then: 変数名と型が含まれる
    assert "foo" in output
    assert "bar" in output


def test_show_vars_empty_when_no_variables(state):
    # given: 変数が何も定義されていない
    # when: SHOW_VARS を呼ぶ
    output = show_vars(state)
    # then: 空である旨のメッセージが返る
    assert "No variables" in output


# =============================================================================
# add_function
# =============================================================================


def test_add_function_is_callable_inside_repl(state):
    # given: Python 関数を add_function で登録している
    add_function(state, "double", lambda x: x * 2)
    # when: REPL 内でその関数を呼び出す
    result = execute_code(state, "print(double(21))")
    # then: 期待通りの結果が stdout に出る
    assert result.stdout == "42\n"


def test_add_function_survives_overwrite_in_repl(state):
    # given: 関数を登録した後、ユーザーコードが同名変数を上書きしている
    add_function(state, "myfn", lambda: "original")
    execute_code(state, "myfn = None")
    # when: 次の実行で再びその関数を呼ぶ
    result = execute_code(state, "print(myfn())")
    # then: scaffold リストアにより元の関数が復活している
    assert result.stdout == "original\n"


def test_add_function_result_accessible_as_variable(state):
    # given: 文字列を返す関数を登録している
    add_function(state, "greet", lambda name: f"Hello, {name}!")
    # when: 関数の戻り値を変数に代入して FINAL_VAR で取り出す
    execute_code(state, "msg = greet('world')")
    r = execute_code(state, "FINAL_VAR('msg')")
    # then: 正しい文字列が final_answer に入る
    assert r.final_answer == "Hello, world!"


# =============================================================================
# context management
# =============================================================================


def test_load_context_string_is_accessible_in_repl(state):
    # given: 文字列コンテキストをロードしている
    load_context(state, "important data")
    # when: REPL 内で context 変数を参照する
    result = execute_code(state, "print(context)")
    # then: ロードした内容が出力される
    assert "important data" in result.stdout


def test_load_context_dict_is_accessible_in_repl(state):
    # given: dict コンテキストをロードしている
    load_context(state, {"key": "value"})
    # when: REPL 内で context["key"] を参照する
    result = execute_code(state, "print(context['key'])")
    # then: 期待通りの値が出力される
    assert result.stdout.strip() == "value"


def test_add_context_with_explicit_index(state):
    # given: index=2 でコンテキストを追加している
    add_context(state, "ctx2", context_index=2)
    # when: REPL 内で context_2 を参照する
    result = execute_code(state, "print(context_2)")
    # then: 追加した内容が出力される
    assert "ctx2" in result.stdout


def test_add_context_auto_increments_index(state):
    # given: コンテキストを2回追加している (index 自動採番)
    add_context(state, "first")
    add_context(state, "second")
    # when: それぞれ参照する
    r0 = execute_code(state, "print(context_0)")
    r1 = execute_code(state, "print(context_1)")
    # then: 順番通りに保存されている
    assert "first" in r0.stdout
    assert "second" in r1.stdout


# =============================================================================
# history management
# =============================================================================


def test_add_history_accessible_as_history_variable(state):
    # given: メッセージ履歴を add_history で登録している
    history = [{"role": "user", "content": "hi"}]
    add_history(state, history)
    # when: REPL 内で history 変数を参照する
    result = execute_code(state, "print(history[0]['role'])")
    # then: 登録した履歴が参照できる
    assert result.stdout.strip() == "user"


def test_add_history_is_deep_copied(state):
    # given: リストを add_history で登録している
    original = [{"role": "user", "content": "hello"}]
    add_history(state, original)
    # when: 元のリストを変更する
    original[0]["content"] = "mutated"
    # then: REPL 内の history は変更されていない
    result = execute_code(state, "print(history[0]['content'])")
    assert "mutated" not in result.stdout


# =============================================================================
# cleanup
# =============================================================================


def test_cleanup_clears_namespace(state):
    # given: REPL に変数が定義されている
    execute_code(state, "x = 99")
    # when: cleanup を呼ぶ
    cleanup(state)
    # then: globals / locals が空になる
    assert state.globals == {}
    assert state.locals == {}
