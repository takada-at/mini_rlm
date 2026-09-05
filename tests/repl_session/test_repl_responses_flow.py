from typing import Any
from unittest.mock import Mock

import pytest

from mini_rlm.llm import create_request_context
from mini_rlm.repl_session.data_model import (
    ReplSessionCommand,
    ReplSessionCommandType,
    ReplSessionLimits,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
)
from mini_rlm.repl_session.executor_command import (
    execute_append_history,
    execute_call_llm,
    execute_compacting,
)
from mini_rlm.repl_session.reducer import reduce_repl_session


def _http_response(body: dict[str, Any]) -> Mock:
    response = Mock()
    response.json.return_value = body
    return response


def test_rlm_http_flow_replays_output_then_compaction_without_repeating_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # テストしたいふるまい: API解析・reducer・圧縮を接続したときに、推論と圧縮項目が次回HTTP入力へ届く。
    # give: 反復応答、圧縮応答、最終応答を返すHTTP境界
    context = create_request_context(
        "https://example.invalid/v1/responses",
        "test",
        api_type="responses",
        request_params={"reasoning": {"effort": "high"}},
    )
    output = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "encrypted_content": "opaque",
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "inspect next", "annotations": []}
            ],
        },
    ]
    compacted = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "task"}],
        },
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "compressed"},
    ]
    request = Mock(
        side_effect=[
            _http_response(
                {
                    "status": "completed",
                    "output": output,
                    "usage": {
                        "total_tokens": 10,
                        "input_tokens": 5,
                        "output_tokens": 5,
                    },
                }
            ),
            _http_response(
                {
                    "object": "response.compaction",
                    "output": compacted,
                    "usage": {
                        "total_tokens": 20,
                        "input_tokens": 10,
                        "output_tokens": 10,
                    },
                }
            ),
            _http_response({"status": "completed", "output": output}),
        ]
    )
    monkeypatch.setattr(context.session, "request", request)
    state = ReplSessionState(
        prompt="task",
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=1000, iteration_limit=5, timeout_seconds=60, error_threshold=2
        ),
        started_at_seconds=0,
        current_time_seconds=1,
    )
    # when: 呼び出し・履歴追加・圧縮・次回呼び出しを既存の実行経路で行う
    result = execute_call_llm(
        ReplSessionCommand(type=ReplSessionCommandType.CALL_LLM), context, state
    )
    called, _ = reduce_repl_session(state, result)
    appended, _ = reduce_repl_session(
        called,
        execute_append_history(
            ReplSessionCommand(type=ReplSessionCommandType.APPEND_HISTORY), called
        ),
    )
    needs_compaction = appended.model_copy(
        update={
            "limits": appended.limits.model_copy(
                update={
                    "context_window_tokens": 1000,
                    "output_token_reserve": 100,
                    "compacting_threshold_rate": 0.1,
                }
            )
        }
    )
    compact_result = execute_compacting(
        ReplSessionCommand(type=ReplSessionCommandType.COMPACTING),
        needs_compaction,
        context,
    )
    compact_state, _ = reduce_repl_session(needs_compaction, compact_result)
    final = execute_call_llm(
        ReplSessionCommand(type=ReplSessionCommandType.CALL_LLM), context, compact_state
    )
    # then: 圧縮入力は元の推論を含み、その後はsystem指示と圧縮結果だけを送る
    assert final.type == ReplSessionResultType.SUCCESS
    assert request.call_args_list[1].args[1].endswith("/responses/compact")
    assert request.call_args_list[1].kwargs["json"]["input"][2:] == output
    assert request.call_args_list[2].kwargs["json"]["input"][1:] == compacted
    assert len(request.call_args_list[2].kwargs["json"]["input"]) == 3
    assert "reasoning" not in request.call_args_list[1].kwargs["json"]
    assert compact_state.total_tokens == 30
    assert compact_state.model_token_usages[0].completion_tokens == 15
