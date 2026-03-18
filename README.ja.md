# mini_rlm

PDF と画像を扱える Recursive Language Model (RLM) の実装です。
通常のテキスト中心の RLM ではなく、OpenAI 互換のマルチモーダル API を前提に、REPL 上で PDF・画像処理を組み合わせて問題を分割しながら解くことを目指しています。

現在のステータスは実装中です。公開 API はまだ安定していません。

## できること

- 画像をそのまま LLM に渡して問い合わせる
- PDF のページを画像化またはテキスト抽出して扱う
- 一時ディレクトリ付きの永続 REPL で複数ターンのコード実行を継続する
- `rlm_query()` による子 REPL の再帰実行でサブ問題を切り出す
- reducer + executor パターンで状態遷移と副作用を分離する

## 全体像

主要なパッケージは次の通りです。

- `mini_rlm/llm`: OpenAI 互換 API へのリクエスト、メッセージ構築、token usage 抽出
- `mini_rlm/repl`: Python REPL の状態管理とコード実行
- `mini_rlm/repl_session`: LLM 呼び出しと REPL 実行を繰り返すセッション制御
- `mini_rlm/recursive_query`: 子 REPL を使った再帰実行
- `mini_rlm/pdf`: PDF ページ数取得、画像変換、テキスト抽出
- `mini_rlm/image`: 画像ファイルの読み込みと `ImageData` 変換
- `mini_rlm/custom_functions`: REPL に注入する関数コレクション

REPL には用途別の関数コレクションを渡せます。

- `minimal_function_collection()`
- `image_function_collection()`
- `pdf_function_collection()`

その他

- `scripts`: 外部提供用のツール群。
    - `scripts/pdf_chapter_split.py`: PDFの章を切り出す。
- `dev_scripts`: 開発用のスクリプト。
- `manual_tests`: 手動実行用の結合テスト

## 設計方針

このリポジトリでは、ドメインごとにパッケージを分け、データ構造は各ドメインの `data_model.py` に置きます。

- データモデル以外のクラスは使わず、関数型志向で実装する
- 複雑な状態遷移は reducer に閉じ込める
- 副作用は executor に寄せ、reducer は純粋関数として保つ
- テスト対象は主に「複雑なロジックを持つが副作用のない部分」に限定する

`repl_session` と `llm` は reducer パターンで実装されています。

## セットアップ

前提:

- Python 3.12+
- `uv`

依存関係をインストールします。

```bash
uv sync --dev
```

## 環境変数

手動実行スクリプトでは次の環境変数を使います。

```bash
export MINI_RLM_LLM_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_API_KEY="..."
export MINI_RLM_LLM_MODEL="gpt-4.1-mini"
```

PDF 用の REPL セッションでは、必要に応じてサブモデル用の設定も使えます。

```bash
export MINI_RLM_LLM_SUB_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_SUB_API_KEY="..."
```

この実装は、`model` と `messages` を JSON body に入れて `POST` する OpenAI 互換 endpoint を想定しています。

## クイックスタート

### ユーザー向け chat CLI を起動する

```bash
export MINI_RLM_LLM_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_API_KEY="..."
export MINI_RLM_LLM_MODEL="gpt-4.1-mini"
uv run mini-rlm chat --file /path/to/book.pdf
```

chat セッション中では `/help`, `/files`, `/add <path>`, `/run <prompt>`, `/exit` が使えます。

### 単発の agent execution を実行する

```bash
export MINI_RLM_LLM_ENDPOINT="https://your-host/v1/chat/completions"
export MINI_RLM_LLM_API_KEY="..."
export MINI_RLM_LLM_MODEL="gpt-4.1-mini"
uv run mini-rlm run --file /path/to/book.pdf --prompt "Find the page where Chapter 2 begins."
```

### PDFの章を切り出す。

```bash
export API_ENDPOINT="https://your-host/v1/chat/completions"
export API_KEY="..."
uv run python scripts/pdf_chapter_split.py <pdf_path> <chapter number(1-index)>
```

### 画像付き REPL セッションを動かす

```bash
uv run python manual_tests/repl_describe_image.py
```

## ライブラリとして使う

最小例です。

```python
from pathlib import Path

from mini_rlm import (
    ReplExecutionRequest,
    ReplSetupRequest,
    create_request_context,
    execute_repl_session,
    pdf_function_collection,
)

request_context = create_request_context(
    endpoint_url="https://your-host/v1/chat/completions",
    api_key="...",
    model="gpt-4.1-mini",
)

result = execute_repl_session(
    ReplExecutionRequest(
        prompt="Find the page where Chapter 2 starts.",
        setup=ReplSetupRequest(
            request_context=request_context,
            file_paths=[Path("/path/to/book.pdf")],
            context_payload={"pdf_path": "book.pdf"},
            functions=pdf_function_collection(),
        ),
        session_request_context=request_context,
    )
)

print(result.termination_reason)
print(result.final_answer)
```

`execute_repl_session()` は REPL のセットアップ、セッション実行、クリーンアップまでまとめて行います。

## 開発コマンド

```bash
make format
make lint
make typecheck
make test
```

## テスト方針

- `pytest` を使う
- テストはドメイン単位で `tests/<domain>/` に置く
- ふるまいベースで `give / when / then` コメントをつける
- 副作用がシンプルなコードや単純なグルーコードには、むやみに単体テストを増やさない

## ディレクトリ例

```text
mini_rlm/
  image/
    data_model.py
    convert.py
  pdf/
    data_model.py
    convert.py
  repl/
  repl_session/
  recursive_query/
  llm/
tests/
  image/
  pdf/
  repl/
  repl_session/
```
