from pathlib import Path

import pytest

from mini_rlm.chat_session.convert import (
    build_forced_run_decision,
    build_run_context_payload,
    convert_paths_to_attachments,
    parse_chat_decision,
    validate_chat_decision,
)
from mini_rlm.chat_session.data_model import ChatDecision, ChatDecisionType


def test_parse_chat_decision_accepts_fenced_json() -> None:
    # テストしたいふるまい: JSON が markdown の fenced code block に包まれている場合でも、
    # 正しく ChatDecision としてパースできること
    # give: JSON が markdown fence に包まれている
    text = """```json
    {
      "type": "respond_chat",
      "message": "hello"
    }
    ```"""
    # when: chat decision を parse する
    decision = parse_chat_decision(text)
    # then: fenced JSON でも正しく ChatDecision になる
    assert decision == ChatDecision(
        type=ChatDecisionType.RESPOND_CHAT,
        message="hello",
    )


def test_validate_chat_decision_rejects_unknown_files() -> None:
    # テストしたいふるまい: ChatDecision が添付されていないファイルを要求している場合、妥当性検証でエラーになること
    # give: 添付されていないファイルを run_agent が要求している
    attachments = convert_paths_to_attachments([Path("book.pdf")])
    decision = ChatDecision(
        type=ChatDecisionType.RUN_AGENT,
        task="Inspect the PDF.",
        reason="The request needs file analysis.",
        file_names=["missing.pdf"],
        success_criteria="Return the answer.",
        user_facing_preamble="Running the agent.",
    )
    # when / then: 妥当性検証でエラーになる
    with pytest.raises(ValueError, match="unknown files"):
        validate_chat_decision(decision, attachments)


def test_build_forced_run_decision_selects_all_attachments() -> None:
    # テストしたいふるまい: build_forced_run_decision が与えられた全ての添付ファイルを選択すること
    # give: 複数添付がある
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("diagram.png")])
    # when: forced run decision を組み立てる
    decision = build_forced_run_decision("Investigate the files.", attachments)
    # then: 全添付が選択される
    assert decision.type == ChatDecisionType.RUN_AGENT
    assert decision.file_names == ["book.pdf", "diagram.png"]


def test_build_run_context_payload_sets_single_pdf_and_image_keys() -> None:
    # テストしたいふるまい: build_run_context_payload が PDF と画像のそれぞれ1つずつの添付ファイルに対して
    # 互換用の pdf_path / image_path キーを正しく設定すること

    # give: PDF と画像がそれぞれ1つずつある
    attachments = convert_paths_to_attachments([Path("book.pdf"), Path("diagram.png")])
    # when: run 用 context payload を組み立てる
    payload = build_run_context_payload(attachments)
    # then: 互換用の pdf_path / image_path も含まれる
    assert payload["attached_files"] == ["book.pdf", "diagram.png"]
    assert payload["pdf_path"] == "book.pdf"
    assert payload["image_path"] == "diagram.png"
