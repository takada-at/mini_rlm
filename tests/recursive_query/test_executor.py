from typing import Any, cast

import requests

from mini_rlm.custom_functions.data_model import FunctionCollection
from mini_rlm.llm.data_model import Endpoint, ModelTokenUsage, RequestContext
from mini_rlm.recursive_query.convert import default_recursive_query_config
from mini_rlm.recursive_query.data_model import RecursiveQueryRequest
from mini_rlm.recursive_query.executor import execute_recursive_query
from mini_rlm.repl.repl import cleanup, create_repl
from mini_rlm.repl_session.data_model import ReplSessionResult, TerminationReason


def test_execute_recursive_query_exposes_model_token_usages(
    monkeypatch,
) -> None:
    # give: 子 REPL セッションがモデル別 usage を返す
    parent_repl_state = create_repl()
    request_context = RequestContext(
        session=cast(Any, requests.Session()),
        endpoint=Endpoint(url="https://example.com/v1/chat/completions"),
        kwargs={"model": "gpt-request"},
    )

    def fake_execute_repl_session(_: Any) -> ReplSessionResult:
        return ReplSessionResult(
            termination_reason=TerminationReason.COMPLETED,
            final_answer="done",
            total_iterations=2,
            total_tokens=30,
            model_token_usages=[
                ModelTokenUsage(
                    model_name="gpt-child",
                    prompt_tokens=20.0,
                    completion_tokens=10.0,
                )
            ],
            total_time_seconds=1.5,
        )

    monkeypatch.setattr(
        "mini_rlm.recursive_query.executor.execute_repl_session",
        fake_execute_repl_session,
    )

    try:
        # when: recursive query を実行する
        result = execute_recursive_query(
            request=RecursiveQueryRequest(prompt="hello"),
            request_context=request_context,
            parent_repl_state=parent_repl_state,
            function_collection=FunctionCollection(functions=[]),
            config=default_recursive_query_config(),
            runtime=None,
        )
    finally:
        cleanup(parent_repl_state)

    # then: 子セッションのモデル別 usage が公開結果に露出される
    assert result.model_token_usages == [
        ModelTokenUsage(
            model_name="gpt-child",
            prompt_tokens=20.0,
            completion_tokens=10.0,
        )
    ]
