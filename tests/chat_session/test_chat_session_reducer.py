from pathlib import Path

from mini_rlm.chat_session.convert import convert_paths_to_attachments
from mini_rlm.chat_session.data_model import (
    ChatDecision,
    ChatDecisionType,
    ChatSessionCommandType,
    ChatSessionResultType,
    ChatSessionState,
    CommandResult,
    RunSummary,
)
from mini_rlm.chat_session.reducer import reduce_chat_session
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
        pending_user_text="Inspect the PDF.",
    )
    return base.model_copy(update=updates)


def test_reduce_chat_session_respond_chat_exits_with_appended_turn() -> None:
    # テストしたいふるまい: reduce_chat_session で respond_chatが成功した場合、チャットのターンが追加され、
    # 次コマンドがCOMPLETE_TURN コマンドになること

    # give: decision command が respond_chat で成功している
    prev_state = _state()
    prev_result = CommandResult(
        command_type=ChatSessionCommandType.DECIDE,
        type=ChatSessionResultType.SUCCESS,
        decision=ChatDecision(
            type=ChatDecisionType.RESPOND_CHAT,
            message="I can answer directly.",
        ),
        consumed_tokens=12,
    )
    # when: reducer を実行する
    next_state, command = reduce_chat_session(prev_state, prev_result)
    # then: turn が追加される。次コマンドは COMPLETE_TURN になる
    assert command.type == ChatSessionCommandType.COMPLETE_TURN
    assert next_state.pending_user_text is None
    assert next_state.turns[-1].assistant_text == "I can answer directly."
    assert next_state.total_tokens == 12


def test_reduce_chat_session_run_agent_decision_advances_to_run_command() -> None:
    # テストしたいふるまい: reduce_chat_session で decide command が 成功し、run_agent の決定を返した場合、
    # 次コマンドが RUN_AGENT になること

    # give: decision command が run_agent で成功している
    prev_state = _state()
    prev_result = CommandResult(
        command_type=ChatSessionCommandType.DECIDE,
        type=ChatSessionResultType.SUCCESS,
        decision=ChatDecision(
            type=ChatDecisionType.RUN_AGENT,
            task="Inspect the PDF.",
            reason="The file must be analyzed.",
            file_names=["book.pdf"],
            success_criteria="Return the answer.",
            user_facing_preamble="Running the agent.",
        ),
    )
    # when: reducer を実行する
    next_state, command = reduce_chat_session(prev_state, prev_result)
    # then: 次コマンドはrun_agentになる
    assert command.type == ChatSessionCommandType.RUN_AGENT
    assert next_state.pending_decision is not None
    assert next_state.pending_decision.file_names == ["book.pdf"]


def test_reduce_chat_session_run_agent_result_exits_with_summary() -> None:
    # テストしたいふるまい: reduce_chat_session で run_agent command が 成功した場合、
    # チャットのターンが追加され、run summary が記録されること。

    # give: run_agent 実行直前の state と、その成功結果がある
    prev_state = _state(
        pending_decision=ChatDecision(
            type=ChatDecisionType.RUN_AGENT,
            task="Inspect the PDF.",
            reason="The file must be analyzed.",
            file_names=["book.pdf"],
            success_criteria="Return the answer.",
            user_facing_preamble="Running the agent.",
        )
    )
    prev_result = CommandResult(
        command_type=ChatSessionCommandType.RUN_AGENT,
        type=ChatSessionResultType.SUCCESS,
        assistant_text="Chapter 2 starts on page 17.",
        run_summary=RunSummary(
            termination_reason="Completed",
            final_answer="Chapter 2 starts on page 17.",
            total_iterations=3,
            total_tokens=120,
            total_time_seconds=1.5,
        ),
        consumed_tokens=120,
    )
    # when: reducer を実行する
    next_state, command = reduce_chat_session(prev_state, prev_result)
    # then: turn を追加して COMPLETE_TURN コマンドになる。run summary が記録される
    assert command.type == ChatSessionCommandType.COMPLETE_TURN
    assert next_state.turns[-1].run_summary is not None
    assert next_state.turns[-1].assistant_text == "Chapter 2 starts on page 17."
    assert next_state.total_tokens == 120
