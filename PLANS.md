# REPL Session Iteration を Reducer パターンで実装する

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

このドキュメントはリポジトリルートの `PLANS.md` であり、`exec-plans` の規約に従って更新し続ける。

## Purpose / Big Picture

この変更により、REPL セッションの 1 イテレーション実行が、純粋関数の reducer と副作用 executor に分離される。ユーザーは「LLM 呼び出し」「コード実行」「履歴更新」「完了判定」を安定した順序で反復でき、トークン上限、反復上限、タイムアウト、エラー閾値、キャンセルを同じ状態遷移モデルで扱える。動作確認は `pytest` の `tests/repl_session` 配下のシナリオテストで行い、成功終了と各失敗終了理由を再現する。

## Progress

- [x] (2026-03-05 06:58Z) 既存コード (`mini_rlm/llm`) の reducer/executor パターンとテストスタイルを調査した。
- [x] (2026-03-05 07:06Z) `repl_session` 実装方針をこの ExecPlan に初版として記述した。
- [x] (2026-03-05 07:00Z) `mini_rlm/repl_session/data_model.py` に状態・コマンド・結果・終了理由モデルを追加した。
- [x] (2026-03-05 07:00Z) `mini_rlm/repl_session/reducer.py` に純粋関数 reducer を実装した。
- [ ] `mini_rlm/repl_session/executor.py` に副作用 executor を実装する。
- [ ] `tests/repl_session/test_executor.py` を追加する。
- [x] (2026-03-05 07:00Z) `tests/repl_session/test_repl_session_reducer.py` を追加し、10件パスを確認した。
- [ ] `make test` を実行し、結果を反映する。

## Surprises & Discoveries

- Observation: 既存の `mini_rlm/llm` は reducer が `prev_state` と `prev_command_result` を受け、`next_state` と `command` を返す形式で統一されている。
  Evidence: `mini_rlm/llm/reducer.py` の `reduce_request` が同署名で実装され、`mini_rlm/llm/executor.py` が同ループで消費している。

- Observation: テストは日本語コメントの give/when/then 形式で統一されている。
  Evidence: `tests/llm/test_reducer.py`, `tests/repl/test_repl.py` の各テスト先頭コメント。

## Decision Log

- Decision: `repl_session` は新規パッケージ `mini_rlm/repl_session` として実装し、既存 `mini_rlm/repl` には直接ロジックを追加しない。
  Rationale: ドメイン分離規約（パッケージはドメイン単位）を守り、既存 REPL 実行基盤とセッション制御を疎結合に保つため。
  Date/Author: 2026-03-05 / Codex

- Decision: 初期状態からの標準フローは `call_llm -> execute_code -> append_history -> check_complete` とし、分岐で `compacting`, `complete`, `exit` を発行する。
  Rationale: ユーザー提示の構想を最小変形で reducer の有限状態遷移に落とし込むため。
  Date/Author: 2026-03-05 / Codex

- Decision: 終了理由は enum で明示し、`EXIT` コマンド時に必ず理由を state に保存する。
  Rationale: 実行結果の可観測性を上げ、テストで期待値を固定しやすくするため。
  Date/Author: 2026-03-05 / Codex

## Outcomes & Retrospective

現時点では計画策定のみ完了。実装と検証は未着手。次のマイルストーンで reducer と executor を実装し、最終的に失敗理由ごとの終了挙動をテストで実証する。

## Context and Orientation

このリポジトリはドメインごとにモジュールが分かれており、データ構造は各パッケージの `data_model.py` に定義する規約がある。既存の reducer パターンは `mini_rlm/llm` で使われている。具体的には以下の構成である。

- `mini_rlm/llm/data_model.py`: 状態 (`RequestState`)、コマンド (`RequestCommand`)、実行結果 (`CommandResult`) を Pydantic モデルで定義。
- `mini_rlm/llm/reducer.py`: 純粋関数 `reduce_request(prev_state, prev_command_result)` を定義。副作用を持たない。
- `mini_rlm/llm/executor.py`: ループで reducer を呼び、返却コマンドに応じて副作用処理（HTTP呼び出し、sleep）を行う。
- `tests/llm/test_reducer.py`, `tests/llm/test_executor.py`: reducer 単体と executor の統合的な挙動をテスト。

今回追加する `repl_session` は、REPL の一連のステップを 1 つの状態機械（state machine）として扱う。「状態機械」とは、現在状態と直前のコマンド実行結果から、次状態と次コマンドを決定する方式を指す。状態機械により、終了条件や例外経路を if 文の散在ではなく、明示的な遷移として管理できる。

## Plan of Work

まず `mini_rlm/repl_session/data_model.py` を作り、以下を定義する。`ReplSessionStatus`（例: `RUNNING`, `COMPLETED`, `FAILED`）、`ReplSessionCommandType`（`CALL_LLM`, `EXECUTE_CODE`, `APPEND_HISTORY`, `CHECK_COMPLETE`, `COMPACTING`, `COMPLETE`, `EXIT`）、`ReplSessionResultType`（各コマンドの成功/失敗/スキップ結果）、`TerminationReason`（`TokenLimitExceeded`, `IterationsExhausted`, `Timeout`, `ErrorThresholdExceeded`, `Cancelled` と `Completed` を含む終了理由）、`ReplSessionState`（反復回数、開始時刻、トークン使用量、連続/累積エラー数、履歴長、上限設定、完了フラグ、終了理由など）、`ReplSessionCommand`、`CommandResult`。ここで「上限設定」は token limit / iteration limit / timeout / error threshold / history length limit を指す。

次に `mini_rlm/repl_session/reducer.py` を実装する。`reduce_repl_session(prev_state, prev_command_result)` は次の規則を持つ。初回（`prev_command_result is None`）は事前上限チェックを行い、問題なければ `CALL_LLM` を返す。各ステップ成功時は標準フローに沿って次コマンドへ進む。`CHECK_COMPLETE` 成功時に完了なら `COMPLETE`、未完了なら次イテレーションへ進む。履歴長が上限超過なら `COMPACTING` を優先し、成功後に `CALL_LLM` に戻す。どの遷移でも上限超過を検知した場合は `EXIT` を返し、`termination_reason` を該当値に設定する。コマンド失敗時はエラー数を更新し、閾値超過で `EXIT(ErrorThresholdExceeded)`、未超過なら再試行戦略に従って次コマンドを決める（初版は同コマンド再実行ではなく `EXIT` にしてもよいが、ここは実装時に明示決定し `Decision Log` を更新する）。

続いて `mini_rlm/repl_session/executor.py` を実装する。`execute_repl_session_loop(initial_state, run_call_llm, run_execute_code, run_append_history, run_check_complete, run_compacting, now_fn)` のように依存関数を注入し、テストで副作用を差し替え可能にする。executor は reducer から返るコマンドを受け、該当ハンドラを呼び `CommandResult` を作って次ループに渡す。`COMPLETE` と `EXIT` は副作用なしで終了し、最終 state を返す。

最後にテストを追加する。`tests/repl_session/test_reducer.py` では遷移網羅（初回遷移、標準フロー、履歴圧縮分岐、各終了理由、エラー閾値）を検証する。`tests/repl_session/test_executor.py` では依存関数をスタブして、ループ全体が期待順序でコマンド実行し、終了時 state が正しいことを検証する。コメントは既存規約に合わせ give/when/then で記述する。

## Concrete Steps

作業ディレクトリは `/Users/takada-at/projects/mini_rlm` を前提とする。以下の順で実行する。

1. 実装ファイル作成

    mkdir -p mini_rlm/repl_session
    touch mini_rlm/repl_session/__init__.py
    touch mini_rlm/repl_session/data_model.py mini_rlm/repl_session/reducer.py mini_rlm/repl_session/executor.py

2. テストファイル作成

    mkdir -p tests/repl_session
    touch tests/repl_session/test_reducer.py tests/repl_session/test_executor.py

3. 整形・静的検査・型検査・テスト

    make format
    make lint
    make type-check
    make test

期待される終端出力は、各コマンドが成功した場合 `make` の各ターゲットが 0 exit code で終了し、`pytest` が `tests/repl_session` の新規テストを含めて全件パスすること。

## Validation and Acceptance

受け入れ条件は次の振る舞いで判定する。

- reducer が初回入力で `CALL_LLM` を返す。
- reducer が成功結果を受けると `CALL_LLM -> EXECUTE_CODE -> APPEND_HISTORY -> CHECK_COMPLETE` を順に返す。
- `history_length > history_limit` のとき `COMPACTING` を返し、圧縮成功後に標準フローへ復帰する。
- `token_limit`, `iteration_limit`, `timeout`, `cancelled`, `error_threshold` いずれかの条件成立時、`EXIT` と対応する `TerminationReason` を返す。
- `CHECK_COMPLETE` の結果が完了なら `COMPLETE`（または `EXIT` with `Completed`）で終了する。
- executor が reducer の返すコマンド順序どおりに依存関数を呼び、最終 state を返す。

テスト観点としては「変更前は `tests/repl_session` が存在しないため失敗し、変更後は新規テストがパスする」ことを確認する。

## Idempotence and Recovery

この計画の手順は冪等に実行できる。`touch` は既存ファイルがあっても破壊しない。テスト失敗時は失敗ログに従い該当モジュールを再編集して再実行する。危険な破壊操作（`git reset --hard` やファイル削除）は使わない。もし設計変更が必要になった場合は、先に `Decision Log` と `Progress` を更新してからコードを更新する。

## Artifacts and Notes

初版時点の調査メモ:

    mini_rlm/llm/reducer.py:
      reduce_request(prev_state, prev_command_result) -> (next_state, command)
    mini_rlm/llm/executor.py:
      while True で reducer を呼び、command.type == EXIT で return
    tests/llm/test_reducer.py:
      give/when/then コメント規約を使用

実装後はここに次を追記する。

- `make lint`, `make type-check`, `make test` の要点ログ
- 代表的なテストケース（例: `TokenLimitExceeded`）の短い実行結果

## Interfaces and Dependencies

この実装では新規外部ライブラリを追加しない。既存の `pydantic` を利用し、`mini_rlm/llm` のモデル定義スタイルに合わせる。

`mini_rlm/repl_session/data_model.py` に最低限以下のインターフェースを定義する。

    class TerminationReason(StrEnum):
        TOKEN_LIMIT_EXCEEDED = "TokenLimitExceeded"
        ITERATIONS_EXHAUSTED = "IterationsExhausted"
        TIMEOUT = "Timeout"
        ERROR_THRESHOLD_EXCEEDED = "ErrorThresholdExceeded"
        CANCELLED = "Cancelled"
        COMPLETED = "Completed"

    class ReplSessionCommandType(StrEnum):
        EXIT = "exit"
        COMPLETE = "complete"
        COMPACTING = "compacting"
        CALL_LLM = "call_llm"
        EXECUTE_CODE = "execute_code"
        APPEND_HISTORY = "append_history"
        CHECK_COMPLETE = "check_complete"

    def reduce_repl_session(
        prev_state: ReplSessionState,
        prev_command_result: CommandResult | None,
    ) -> tuple[ReplSessionState, ReplSessionCommand]:

    def execute_repl_session_loop(
        initial_state: ReplSessionState,
        run_call_llm: Callable[[ReplSessionState], CommandResult],
        run_execute_code: Callable[[ReplSessionState], CommandResult],
        run_append_history: Callable[[ReplSessionState], CommandResult],
        run_check_complete: Callable[[ReplSessionState], CommandResult],
        run_compacting: Callable[[ReplSessionState], CommandResult],
        now_fn: Callable[[], float],
    ) -> ReplSessionState

Plan revision note (2026-03-05): 初版作成。ユーザー要求（`repl_session` の reducer パターン設計）を実装可能な粒度に落とし込み、必須セクションと検証手順を追加した。
Plan revision note (2026-03-05): `data_model` と `reducer` 実装、および reducer テスト追加に合わせて Progress を更新した。`type-check` は Makefile で `typecheck` ターゲット名だったため、その実行結果を反映した。
