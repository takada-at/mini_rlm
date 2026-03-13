from pathlib import Path

import pytest

from mini_rlm.recursive_query.convert import (
    build_child_recursive_query_runtime,
    build_child_repl_limits,
    default_recursive_query_config,
    extract_inherited_context_payload,
    list_inherited_file_paths,
    resolve_recursive_query_runtime,
)
from mini_rlm.recursive_query.data_model import RecursiveQueryRuntime


def test_resolve_recursive_query_runtime_uses_default_depth_when_runtime_missing() -> (
    None
):
    # given: runtime が未指定で固定 config がある
    config = default_recursive_query_config()
    # when: runtime を解決する
    runtime = resolve_recursive_query_runtime(None, config)
    # then: max_depth を remaining_depth にした runtime が得られる
    assert runtime.remaining_depth == config.max_depth


def test_build_child_recursive_query_runtime_decrements_remaining_depth() -> None:
    # given: remaining_depth が 2 の runtime がある
    runtime = RecursiveQueryRuntime(remaining_depth=2)
    # when: child runtime を構築する
    child_runtime = build_child_recursive_query_runtime(runtime)
    # then: remaining_depth が 1 減る
    assert child_runtime.remaining_depth == 1


def test_build_child_recursive_query_runtime_raises_when_depth_exhausted() -> None:
    # given: remaining_depth が尽きた runtime がある
    runtime = RecursiveQueryRuntime(remaining_depth=0)
    # when / then: child runtime 構築で例外になる
    with pytest.raises(ValueError, match="max_depth exceeded"):
        _ = build_child_recursive_query_runtime(runtime)


def test_build_child_repl_limits_returns_a_copy() -> None:
    # given: 固定 child limits を持つ config がある
    config = default_recursive_query_config()
    # when: child limits を構築する
    child_limits = build_child_repl_limits(config)
    # then: config の固定値から ReplSessionLimits が組み立てられる
    assert child_limits.token_limit == config.child_token_limit
    assert child_limits.iteration_limit == config.child_iteration_limit
    assert child_limits.timeout_seconds == config.child_timeout_seconds
    assert child_limits.error_threshold == config.child_error_threshold
    assert (
        child_limits.compacting_threshold_rate == config.child_compacting_threshold_rate
    )


def test_list_inherited_file_paths_returns_only_top_level_files(
    tmp_path: Path,
) -> None:
    # given: 親 temp dir にファイルとサブディレクトリがある
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_file = nested_dir / "ignored.txt"
    file_a.write_text("a")
    file_b.write_text("b")
    nested_file.write_text("nested")
    # when: 継承ファイル一覧を取得する
    paths = list_inherited_file_paths(str(tmp_path), inherit_parent_files=True)
    # then: 直下ファイルのみが返る
    assert paths == [file_a, file_b]


def test_list_inherited_file_paths_returns_empty_when_inheritance_disabled(
    tmp_path: Path,
) -> None:
    # given: 親 temp dir にファイルがある
    file_a = tmp_path / "a.txt"
    file_a.write_text("a")
    # when: 継承を無効にして一覧を取得する
    paths = list_inherited_file_paths(str(tmp_path), inherit_parent_files=False)
    # then: 空配列になる
    assert paths == []


def test_extract_inherited_context_payload_returns_context_0_value() -> None:
    # given: 親 locals に context_0 がある
    parent_locals = {"context_0": {"chapter": "intro"}, "context": "ignored"}
    # when: 継承 context を取り出す
    context_payload = extract_inherited_context_payload(parent_locals)
    # then: context_0 が返る
    assert context_payload == {"chapter": "intro"}


def test_extract_inherited_context_payload_returns_none_for_unsupported_type() -> None:
    # given: 親 locals の context_0 が非JSON型である
    parent_locals = {"context_0": object()}
    # when: 継承 context を取り出す
    context_payload = extract_inherited_context_payload(parent_locals)
    # then: 継承しない
    assert context_payload is None
