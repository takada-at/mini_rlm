from pathlib import Path

from mini_rlm.chat_session.convert import convert_paths_to_attachments
from mini_rlm.cli.convert import (
    build_run_prompt,
    parse_chat_input,
    resolve_run_mode,
    select_function_collection,
)
from mini_rlm.cli.data_model import ChatCLIInputType, RunMode


def test_resolve_run_mode_keeps_auto_for_mixed_pdf_and_image_attachments() -> None:
    # テストしたいふるまい: resolve_run_mode が AUTO モードで PDF と画像が混在している場合、
    # 片方のプリセットに潰さず AUTO のままにすること

    # give: PDF と画像が両方添付されている
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("diagram.png")])
    # when: resolve_run_modeでモードを解決する
    mode = resolve_run_mode(RunMode.AUTO, attachments)
    # then: mixed attachments のまま扱えるよう AUTO のまま残る
    assert mode == RunMode.AUTO


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


def test_select_function_collection_returns_mixed_tools_for_auto_mode() -> None:
    # テストしたいふるまい: select_function_collection が AUTO モードで PDF と画像が混在している場合、
    # 両方の helper を含む function collection を返すこと

    # give: PDF と画像が両方添付されている
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("diagram.png")])
    # when: auto mode の function collection を選ぶ
    functions = select_function_collection(RunMode.AUTO, attachments)
    function_names = [function.name for function in functions.functions]
    # then: PDF helper と image helper の両方が含まれる
    assert "convert_pdf_page_to_text" in function_names
    assert "open_image_data" in function_names


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


def test_parse_chat_input_recognizes_run_command_and_forces_agent_execution() -> None:
    # テストしたいふるまい: parse_chat_input が /run コマンドを解釈する際、
    # 実際のプロンプトだけを取り出し、force_run を有効化すること

    # give: /run 付きの chat 入力がある
    # when: CLI 入力を解釈する
    command = parse_chat_input("/run   inspect the attachment  ")
    # then: 通常メッセージ扱いではなく、強制実行付きの prompt になる
    assert command.type == ChatCLIInputType.SEND_MESSAGE
    assert command.message == "inspect the attachment"
    assert command.force_run is True


def test_parse_chat_input_returns_invalid_for_add_without_path() -> None:
    # テストしたいふるまい: parse_chat_input が /add の引数不足を検知し、
    # 添付処理ではなく usage エラーとして返すこと

    # give: パスなしの /add コマンドがある
    # when: CLI 入力を解釈する
    command = parse_chat_input("/add")
    # then: invalid command として usage を返す
    assert command.type == ChatCLIInputType.INVALID
    assert command.error_message == "Usage: /add <path>"


def test_parse_chat_input_accepts_quoted_add_path() -> None:
    # テストしたいふるまい: parse_chat_input が /add の quoted path を解釈する際、
    # 引用符を除去して実際のファイルパスとして扱うこと

    # give: single quote で囲まれた /add path がある
    # when: CLI 入力を解釈する
    command = parse_chat_input("/add '/path/to/book.pdf'")
    # then: quoted path ではなく、実パスとして正規化される
    assert command.type == ChatCLIInputType.ADD_FILE
    assert command.file_path == Path("/path/to/book.pdf")


def test_parse_chat_input_treats_unknown_slash_input_as_chat_message() -> None:
    # テストしたいふるまい: parse_chat_input が未知のスラッシュ始まり入力を見ても、
    # 内部コマンドに誤認せず通常の chat prompt として扱うこと

    # give: 未知の /command 風入力がある
    # when: CLI 入力を解釈する
    command = parse_chat_input("/summarize this pdf")
    # then: ユーザーの prompt としてそのまま渡される
    assert command.type == ChatCLIInputType.SEND_MESSAGE
    assert command.message == "/summarize this pdf"
    assert command.force_run is False
