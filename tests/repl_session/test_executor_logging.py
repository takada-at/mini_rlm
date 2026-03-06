import importlib
import logging
from pathlib import Path

import requests

from mini_rlm.llm.data_model import Endpoint, RequestContext
from mini_rlm.repl.repl import create_repl
from mini_rlm.repl_session.data_model import (
    CommandResult,
    ReplSessionCommand,
    ReplSessionCommandType,
    ReplSessionResultType,
    ReplSessionState,
    ReplSessionStatus,
)


def test_execute_repl_session_loop_writes_debug_log(monkeypatch, tmp_path) -> None:
    # give: ログファイル出力先を一時パスにし、executorを再ロードする
    log_path = tmp_path / "mini_rlm_debug.log"
    monkeypatch.setenv("MINI_RLM_LOG_FILE", str(log_path))

    logger = logging.getLogger("mini_rlm")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    executor = importlib.import_module("mini_rlm.repl_session.executor")
    executor = importlib.reload(executor)

    call_count = {"count": 0}

    def fake_reduce_repl_session(
        state: ReplSessionState,
        prev_result: CommandResult | None,
    ) -> tuple[ReplSessionState, ReplSessionCommand]:
        if call_count["count"] == 0:
            call_count["count"] += 1
            return state, ReplSessionCommand(type=ReplSessionCommandType.CALL_LLM)

        assert prev_result is not None
        assert prev_result.command_type == ReplSessionCommandType.CALL_LLM
        completed_state = state.model_copy(
            update={
                "status": ReplSessionStatus.COMPLETED,
                "is_complete": True,
            }
        )
        return completed_state, ReplSessionCommand(type=ReplSessionCommandType.COMPLETE)

    def fake_execute_call_llm(
        command: ReplSessionCommand,
        request_context: RequestContext,
        session_state: ReplSessionState,
    ) -> CommandResult:
        return CommandResult(
            command_type=command.type,
            type=ReplSessionResultType.SUCCESS,
            last_llm_message="hello",
            consumed_tokens=12,
        )

    monkeypatch.setattr(executor, "reduce_repl_session", fake_reduce_repl_session)
    monkeypatch.setattr(executor, "execute_call_llm", fake_execute_call_llm)

    repl = create_repl()
    request_context = RequestContext(
        session=requests.Session(),
        endpoint=Endpoint(url="http://example.com"),
    )

    # when: repl_sessionループを実行する
    final_state = executor.execute_repl_session_loop(
        repl=repl,
        request_context=request_context,
        prompt="test prompt",
    )

    # then: ログファイルにデバッグログが出力される
    assert final_state.status == ReplSessionStatus.COMPLETED
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "repl_session.start" in log_text
    assert "repl_session.command type=call_llm" in log_text
    assert "repl_session.result command=call_llm" in log_text
    assert "repl_session.end status=completed" in log_text


def test_importing_executor_does_not_initialize_logger(monkeypatch) -> None:
    # give: logger初期化が呼ばれると失敗する状態にする
    import mini_rlm.debug_logger as debug_logger

    def fail_get_logger() -> logging.Logger:
        raise AssertionError("get_logger should not run during import")

    monkeypatch.setattr(debug_logger, "get_logger", fail_get_logger)

    # when: executorモジュールを再ロードする
    executor = importlib.import_module("mini_rlm.repl_session.executor")
    executor = importlib.reload(executor)

    # then: import時にはlogger初期化が走らない
    assert executor.__name__ == "mini_rlm.repl_session.executor"


def test_execute_repl_session_loop_continues_when_logger_setup_fails(
    monkeypatch,
) -> None:
    # give: logger初期化が毎回失敗しても、セッション自体は完了できるようにする
    logger = logging.getLogger("mini_rlm")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    executor = importlib.import_module("mini_rlm.repl_session.executor")
    executor = importlib.reload(executor)

    call_count = {"count": 0}

    def fake_reduce_repl_session(
        state: ReplSessionState,
        prev_result: CommandResult | None,
    ) -> tuple[ReplSessionState, ReplSessionCommand]:
        if call_count["count"] == 0:
            call_count["count"] += 1
            return state, ReplSessionCommand(type=ReplSessionCommandType.CALL_LLM)

        assert prev_result is not None
        completed_state = state.model_copy(
            update={
                "status": ReplSessionStatus.COMPLETED,
                "is_complete": True,
            }
        )
        return completed_state, ReplSessionCommand(type=ReplSessionCommandType.COMPLETE)

    def fake_execute_call_llm(
        command: ReplSessionCommand,
        request_context: RequestContext,
        session_state: ReplSessionState,
    ) -> CommandResult:
        return CommandResult(
            command_type=command.type,
            type=ReplSessionResultType.SUCCESS,
            last_llm_message="hello",
            consumed_tokens=12,
        )

    def fail_get_logger() -> logging.Logger:
        raise OSError("read-only file system")

    monkeypatch.setattr(executor, "reduce_repl_session", fake_reduce_repl_session)
    monkeypatch.setattr(executor, "execute_call_llm", fake_execute_call_llm)
    monkeypatch.setattr(executor, "get_logger", fail_get_logger)
    monkeypatch.setattr(executor, "get_log_file_path", lambda: Path("/tmp/unwritable.log"))

    repl = create_repl()
    request_context = RequestContext(
        session=requests.Session(),
        endpoint=Endpoint(url="http://example.com"),
    )

    # when: logger初期化に失敗する状態でrepl_sessionループを実行する
    final_state = executor.execute_repl_session_loop(
        repl=repl,
        request_context=request_context,
        prompt="test prompt",
    )

    # then: デバッグログが無効化されてもセッションは完了する
    assert final_state.status == ReplSessionStatus.COMPLETED
