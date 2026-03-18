from pathlib import Path

from mini_rlm.chat_session.convert import convert_paths_to_attachments
from mini_rlm.cli.convert import (
    build_run_prompt,
    resolve_run_mode,
    select_function_collection,
)
from mini_rlm.cli.data_model import RunMode


def test_resolve_run_mode_prefers_pdf_when_pdf_is_attached() -> None:
    # テストしたいふるまい: resolve_run_mode が AUTO モードのとき、PDF が添付されている場合は PDF ツールプリセットを選ぶこと

    # give: PDF と画像が両方添付されている
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("diagram.png")])
    # when: auto mode を解決する
    mode = resolve_run_mode(RunMode.AUTO, attachments)
    # then: PDF tool preset が選ばれる
    assert mode == RunMode.PDF


def test_select_function_collection_returns_image_tools_for_image_mode() -> None:
    # テストしたいふるまい: select_function_collection が IMAGE モードのとき、画像向けの function collection を返すこと

    # give: 画像添付がある
    attachments = convert_paths_to_attachments([Path("diagram.png")])
    # when: image mode の function collection を選ぶ
    functions = select_function_collection(RunMode.IMAGE, attachments)
    function_names = [function.name for function in functions.functions]
    # then: 画像向け helper が含まれる
    assert "open_image_data" in function_names
    assert "llm_image_query" in function_names


def test_build_run_prompt_appends_attachment_summary() -> None:
    # テストしたいふるまい: build_run_prompt が CLI 用のプロンプトを構築する際、添付ファイルの要約をプロンプトに追加すること

    # give: 添付ファイルがある
    attachments = convert_paths_to_attachments([Path("book.pdf")])
    # when: CLI の run prompt を組み立てる
    prompt = build_run_prompt("Inspect the PDF.", attachments)
    # then: 添付一覧が追加される
    assert "Inspect the PDF." in prompt
    assert "Attached files already available" in prompt
    assert "- book.pdf (pdf)" in prompt
