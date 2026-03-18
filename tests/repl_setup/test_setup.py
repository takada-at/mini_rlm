from pathlib import Path

from mini_rlm.custom_functions import minimal_function_collection
from mini_rlm.llm import create_request_context
from mini_rlm.repl import cleanup
from mini_rlm.repl_setup import ReplFileRef, setup_repl


def test_setup_repl_accepts_custom_target_names_for_duplicate_basenames(
    tmp_path: Path,
) -> None:
    # テストしたいふるまい: setup_repl が custom target name を受け取り、
    # basename が同じ複数ファイルでも REPL working directory に共存させられること

    # give: basename が同じ 2 つのファイルと、一意な target name がある
    first_dir = tmp_path / "dir_a"
    second_dir = tmp_path / "dir_b"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "report.pdf"
    second_path = second_dir / "report.pdf"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    request_context = create_request_context(
        endpoint_url="https://example.invalid/v1/chat/completions",
        api_key="dummy",
        model="gpt-test",
    )

    # when: custom target name を指定して REPL を初期化する
    repl_context = setup_repl(
        request_context=request_context,
        files=[
            ReplFileRef(source_path=first_path, target_name="report__a.pdf"),
            ReplFileRef(source_path=second_path, target_name="report__b.pdf"),
        ],
        functions=minimal_function_collection(),
    )

    # then: REPL working directory に両方のファイルが存在する
    try:
        repl_dir = Path(repl_context.repl_state.temp_dir)
        assert (repl_dir / "report__a.pdf").is_file()
        assert (repl_dir / "report__b.pdf").is_file()
    finally:
        cleanup(repl_context.repl_state)
