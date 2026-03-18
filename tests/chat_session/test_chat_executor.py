from pathlib import Path

from mini_rlm.chat_session import convert_paths_to_attachments, create_chat_session
from mini_rlm.chat_session.executor import _select_function_collection, add_attachment
from mini_rlm.llm import create_request_context


def _create_state():
    request_context = create_request_context(
        endpoint_url="https://example.invalid/v1/chat/completions",
        api_key="dummy",
        model="gpt-test",
    )
    return create_chat_session(chat_request_context=request_context)


def test_select_function_collection_keeps_pdf_and_image_tools_for_mixed_files() -> None:
    # テストしたいふるまい: PDF と画像が混在する場合でも、
    # chat session の agent run が両方の tool を使える function collection を選ぶこと

    # give: PDF と画像が両方添付されている
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("photo.png")])
    # when: function collection を選ぶ
    functions = _select_function_collection(attachments)
    function_names = [function.name for function in functions.functions]
    # then: PDF helper と image helper の両方が含まれる
    assert "convert_pdf_page_to_text" in function_names
    assert "open_image_data" in function_names


def test_add_attachment_disambiguates_duplicate_basenames_incrementally(
    tmp_path: Path,
) -> None:
    # テストしたいふるまい: /add を繰り返して basename が同じファイルを追加しても、
    # attachment 名が再計算されて衝突しないこと

    # give: basename が同じ 2 つの実ファイルが別ディレクトリにある
    first_dir = tmp_path / "dir_a"
    second_dir = tmp_path / "dir_b"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "report.pdf"
    second_path = second_dir / "report.pdf"
    first_path.write_text("first", encoding="utf-8")
    second_path.write_text("second", encoding="utf-8")
    state = _create_state()

    # when: 1 件ずつ attachment を追加する
    state = add_attachment(state, first_path)
    state = add_attachment(state, second_path)

    # then: basename が同じでも、最終 state では一意な attachment 名になる
    assert [attachment.path for attachment in state.attachments] == [
        first_path,
        second_path,
    ]
    assert len({attachment.name for attachment in state.attachments}) == 2
