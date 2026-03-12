# ADR 0001: Recursive `rlm_query` Constraints

## Status

Accepted

## Context

`rlm_query` を REPL 内で提供するにあたり、再帰実行の自由度を広く取りすぎると、状態共有、予算管理、実行コストの制御が難しくなる。

特に以下の問題がある。

- 親 REPL と child REPL の state 共有は deadlock や状態汚染の原因になる
- 再帰深さが動的だと、モデルが過剰に分割しやすい
- child session の limits を呼び出し側ごとに変えると、コスト見積もりと運用が不安定になる

## Decision

以下を `rlm_query` の規定とする。

- child REPL は毎回新規作成する
- `max_depth` は固定値で運用する
- child limits も固定値で運用する
- shared usage ledger を導入し、REPL helper 経由の LLM token usage を集計対象に含める

## Consequences

### Positive

- 親子の REPL state が分離され、再入による問題を避けやすい
- 再帰コストの上限が読みやすくなる
- helper 経由の LLM 使用量も session 側で追跡しやすくなる

### Negative

- child REPL の都度初期化コストがかかる
- `max_depth` と child limits を柔軟に最適化する余地は減る
- shared usage ledger の導入により REPL 実行経路に追加の配線が必要になる

## Implementation Notes

- Phase 1 では `FunctionFactoryContext` を導入し、factory が `request_context` 以外の実行時情報も受け取れるようにする
- Phase 2 では shared usage ledger を導入し、`execute_code` 単位で helper 呼び出しの token usage 差分を回収できるようにする
