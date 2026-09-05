# Responses API対応後のレビュー記録

- 記録日: 2026-09-05
- 状態: 未対応
- 対象: Responses API対応・推論履歴保持の実装と、その周辺の既存処理

## 概要

実装後のレビューで、再試行・トークン集計・履歴圧縮について以下の懸念を確認した。修正方針は提案であり、この記録時点では実装していない。

| 優先度 | 問題 | 区分 |
| --- | --- | --- |
| P1 | 未完了応答を同じ設定で最大25回再試行する | 今回の対応で要修正 |
| P1 | 失敗した応答のusageを消費トークンに計上しない | 今回の対応で要修正 |
| P2 | 圧縮判定に現在の履歴サイズではなく累積消費量を使っている | 既存設計の課題 |

## 1. 未完了応答の再試行が二重になる（P1）

### 現象と原因

`parse_response` は `status != "completed"` の応答をエラーにする。LLM executorはこれを `INVALID_RESPONSE` に変換し、API層が同じペイロードで最大5回再試行する。

API層の試行が尽きると、`execute_call_llm` がRLMセッションのエラー結果を返す。セッションreducerも同じコマンドを再試行するため、既定の `error_threshold=5` では、タイムアウトなどで先に終了しない限り最大25回のAPIリクエストになる。

出力上限による未完了や設定不備など、同じ設定の再送では改善しない失敗にも再試行が適用される。待ち時間と不要な生成処理が増える。

### 関連コード

- `mini_rlm/llm/protocol.py`: `parse_response`
- `mini_rlm/llm/executor.py`: `_run_request_command`
- `mini_rlm/llm/reducer.py`: `_is_retryable`
- `mini_rlm/llm/api_request.py`: `run_api_request` のリトライ設定
- `mini_rlm/repl_session/executor_command.py`: `execute_call_llm`
- `mini_rlm/repl_session/reducer.py`: `reduce_repl_session`

### 修正方針案

- 一時的な通信障害と、出力上限・設定不備などの失敗を分類する。
- 再試行可能かどうかをセッション側まで保持し、API層で再試行しない失敗を外側で再試行しないようにする。
- 出力上限による未完了は、明示的に終了するか設定を変更して再開する方針を決める。同じ設定での自動再送を繰り返さない。

## 2. 失敗した応答のトークン消費を集計しない（P1）

### 現象と原因

応答解析に失敗すると、LLM executorはエラー種別と文字列だけを返し、受信したレスポンスのusageを保持しない。RLMセッションへ返すエラー結果も `consumed_tokens=0` のままになる。

そのため、API応答に消費量が記録されていても、セッションのトークン表示と上限判定には反映されない。再試行後に成功した場合も、失敗した試行の消費量が欠落する。

### 関連コード

- `mini_rlm/llm/executor.py`: 応答解析失敗時の `CommandResult` 生成
- `mini_rlm/llm/data_model.py`: リクエスト結果・状態のデータモデル
- `mini_rlm/repl_session/executor_command.py`: `execute_call_llm` のエラー結果
- `mini_rlm/llm/token_usage.py`: usageの変換・集計

### 修正方針案

- 応答本文の解析成否とusageの回収を分離する。
- 各試行で取得できたusageを累積し、成功・失敗のどちらでも呼び出し元へ渡す。
- API層とセッション層の間で二重計上しないよう、消費量を受け渡す単位を明確にする。
- レスポンスを取得できない通信障害については、消費量不明と実際のゼロを区別する。

## 1・2の再現結果

実APIへの接続は行わず、HTTP送信境界をモックに置き換えて確認した。毎回、次の内容を持つ応答を返した。

```json
{
  "status": "incomplete",
  "incomplete_details": {"reason": "max_output_tokens"},
  "output": [],
  "usage": {
    "input_tokens": 10,
    "output_tokens": 20,
    "total_tokens": 30
  }
}
```

API層の待機を無効化し、セッションの `error_threshold=5`、十分な時間・トークン予算で `reduce_repl_session` と `execute_call_llm` を繰り返した。

| 観測項目 | 結果 |
| --- | --- |
| HTTP送信回数 | 25回 |
| モック応答のusage合計 | 750トークン |
| セッションの記録 | 0トークン |
| 終了理由 | `ErrorThresholdExceeded` |

750トークンはモック応答の合計であり、実際の課金実績ではない。実装時の147テストは成功していたが、この失敗・再試行・集計の組み合わせは検出できていなかった。

## 3. 圧縮判定値と実際の履歴サイズが一致しない（P2）

### 現象と原因

`_apply_result` は各コマンドの `consumed_tokens` を `current_history_tokens` に加算している。この値には、同じ履歴を毎回再送した入力トークンや、コード実行中のサブクエリの消費量も含まれる。

さらに圧縮閾値を消費予算の `token_limit` から算出し、圧縮後は `current_history_tokens` を0に戻している。現在の入力サイズやモデルのコンテキスト容量を直接表す値ではないため、圧縮が早すぎたり、容量超過を防げなかったりする可能性がある。

これは今回の変更以前から存在する設計上の課題であり、コードの確認による指摘。実APIでのコンテキスト容量超過は再現していない。

### 関連コード

- `mini_rlm/repl_session/reducer.py`: `_apply_result` と圧縮成功時の状態更新
- `mini_rlm/repl_session/data_model.py`: `is_compaction_limit_exceeded`
- `mini_rlm/repl_session/executor_command.py`: サブクエリを含むコード実行の消費量集計

### 修正方針案

- 累積消費予算と、現在の履歴サイズ・コンテキスト容量を別の状態・設定として管理する。
- 履歴サイズの判定に、過去の再送分や独立したサブクエリの消費量を加算しない。
- 圧縮後の履歴もサイズを持つものとして扱い、出力用の余裕を確保する。

## 対応順序と検証候補

まず1と2を一緒に修正し、その後に3の予算・容量管理を整理する。

- 出力上限による未完了を同じ設定で繰り返し送信しない。
- 再試行不能なエラーをセッション側で再試行しない。
- 失敗後に成功した場合と、最後まで失敗した場合の両方で、取得済みusageを一度だけ計上する。
- 同じ履歴の再送やサブクエリの消費によって、現在の履歴サイズを過大評価しない。
- 圧縮後も履歴サイズと累積消費予算をそれぞれ正しく保持する。
