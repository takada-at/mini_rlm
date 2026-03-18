from pathlib import Path

from mini_rlm.chat_session.convert import build_run_prompt, convert_paths_to_attachments
from mini_rlm.chat_session.data_model import (
    ChatDecision,
    ChatDecisionType,
    ChatSessionState,
    ChatTurn,
)
from mini_rlm.llm import create_request_context


def _state(**updates: object) -> ChatSessionState:
    request_context = create_request_context(
        endpoint_url="https://example.invalid/v1/chat/completions",
        api_key="dummy",
        model="gpt-test",
    )
    base = ChatSessionState(
        chat_request_context=request_context,
        run_request_context=request_context,
        attachments=convert_paths_to_attachments([Path("book.pdf")]),
        turns=[
            ChatTurn(
                user_text="What is in the file?",
                assistant_text="I need to inspect it first.",
                decision_type=ChatDecisionType.RESPOND_CHAT,
            )
        ],
        pending_user_text="Find the page where Chapter 2 starts.",
    )
    return base.model_copy(update=updates)


def test_build_run_prompt_includes_history_current_message_and_success_criteria() -> (
    None
):
    # テストしたいふるまい: build_run_prompt が、会話履歴、現在のユーザーメッセージ、成功基準を含むプロンプトを構築すること
    # give: 会話履歴と現在ターン、success criteria がある
    state = _state()
    decision = ChatDecision(
        type=ChatDecisionType.RUN_AGENT,
        task="Inspect the PDF and find the starting page of Chapter 2.",
        reason="The PDF must be analyzed.",
        file_names=["book.pdf"],
        success_criteria="Return the page number as an integer.",
        user_facing_preamble="Inspecting the PDF.",
    )
    # when: agent run 用 prompt を構築する
    prompt = build_run_prompt(state, decision)
    # then: 実行に必要な文脈が含まれる
    assert "Recent chat history:" in prompt
    assert "What is in the file?" in prompt
    assert "Find the page where Chapter 2 starts." in prompt
    assert "Return the page number as an integer." in prompt
    assert "- book.pdf" in prompt
