# Chat 中の agent run 起動判定を明示的な LLM 判断として実装する

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

この文書自身が repository root の `PLANS.md` であり、以後この作業を進める人はこのファイルを更新し続けること。過去の会話や別ドキュメントを知らなくても、このファイルと現在の working tree だけで作業を再開できる状態を保つ。

## Purpose / Big Picture

この変更の目的は、ユーザーとの Chat の途中で「ここはそのまま返答するべきか」「ここは別の `agent run` に切り出すべきか」を、上位の会話オーケストレータが明示的に判断できるようにすることにある。変更後は、単純な質問にはその場で返答し、PDF や画像や長い文脈の調査のように独立した探索が必要な場面では、LLM が明示的に `agent run` を要求し、その要求をシステムが予算と安全条件を確認したうえで実行する。

ユーザーに見える効果は二つある。第一に、簡単な会話で無駄に `agent run` を起動しなくなる。第二に、`agent run` が必要な場面では「なぜ起動したのか」が会話オーケストレータの決定として説明可能になる。実装完了後は、手元の CLI で一つの Chat turn を実行し、簡単な算術プロンプトでは `agent run` が 0 回、ファイル調査プロンプトでは `agent run` が 1 回以上実行されることを確認できるようにする。

## Progress

- [x] (2026-03-18 07:38Z) `Chat`、`agent run`、`REPL` を別概念として扱う方針を確定した。
- [x] (2026-03-18 07:38Z) `agent run` の起動判定は隠れた heuristic ではなく、LLM による明示的な要求として扱う方針を確定した。
- [x] (2026-03-18 07:38Z) 現状コードの調査を行い、再帰実行が現在は `rlm_query` による REPL 内 delegation として実装されていること、LLM 層には provider-native の tool calling がまだ無いことを確認した。
- [x] (2026-03-18 07:38Z) この `PLANS.md` を初版として作成した。
- [ ] Chat レイヤ専用の `agent_run` インターフェースを追加し、既存の REPL 実行詳細をその裏に隠す。
- [ ] `chat_session` ドメインを追加し、1 Chat turn の reducer / executor を実装する。
- [ ] LLM が `reply` か `agent_run` を厳密な JSON で返す planner prompt と parser を実装する。
- [ ] 予算超過、同一 prompt の連打、decision の malformed JSON を抑止する guardrail を実装する。
- [ ] 単体テストと manual test を追加し、`make format`, `make lint`, `make typecheck`, `make test` を通す。

## Surprises & Discoveries

- Observation: 現在の repository には Chat オーケストレータが存在せず、再帰実行は REPL の中から `rlm_query` を呼ぶ形でのみ提供されている。
  Evidence: `mini_rlm/custom_functions/functions.py` の `create_rlm_query()` は `mini_rlm/recursive_query/executor.py` の `execute_recursive_query()` を直接呼び出している。

- Observation: 既存の LLM API 層は `messages` を POST body に詰めて送るだけで、provider-native の `tools` や `tool_choice` を扱っていない。
  Evidence: `mini_rlm/llm/api_request.py` の `run_api_request()` は `request_body` に `messages` と `context.kwargs` を入れるだけであり、tool calling 固有の field は持たない。

- Observation: 既存の recursive query 実装には、深さ制限、child token 制限、iteration 制限、file 継承など、Chat レイヤにも再利用したい hard gate がすでにある。
  Evidence: `mini_rlm/recursive_query/data_model.py` の `RecursiveQueryConfig` と `mini_rlm/recursive_query/convert.py` の `build_child_repl_limits()` および `build_child_recursive_query_runtime()`。

## Decision Log

- Decision: この作業では `Chat`、`agent run`、`REPL` を明確に分離した用語で扱う。
  Rationale: ユーザーとの会話と Python 実行環境は同じものではない。概念を混同すると API 設計が REPL 中心になり、Chat 中にいつ delegation するかという本来の問題が見えなくなる。
  Date/Author: 2026-03-18 / Codex

- Decision: `agent run` の起動は、上位の Chat オーケストレータに対する LLM の明示的な要求として表現する。
  Rationale: 何が self-contained な subtask かはモデル自身がもっとも判断しやすい。一方で予算、深さ、同一 prompt 連打防止のような hard gate は deterministic に executor 側で扱うべきである。
  Date/Author: 2026-03-18 / Codex

- Decision: Phase 1 では provider-native の tool calling ではなく、厳密な JSON schema を返す planner response で `reply` と `agent_run` を表現する。
  Rationale: 現在の `mini_rlm/llm` は native tool calling を持たず、それを同時に実装するとスコープが広がりすぎる。厳密 JSON でも「LLM が明示的に呼び出す」という要件は満たせる。
  Date/Author: 2026-03-18 / Codex

- Decision: 既存の `rlm_query` は当面残し、Chat レイヤの `agent run` は別経路として追加する。
  Rationale: 既存 manual test と recursive query の動作を壊さずに、新しい Chat → agent run の経路を追加する方が安全である。最初の段階で全面移行は行わない。
  Date/Author: 2026-03-18 / Codex

- Decision: planner の出力は「JSON を含む自由文」ではなく「JSON そのもの」を要求し、parser は余計な文字列を許容しない。
  Rationale: JSON の切り出し heuristic を入れると実装が曖昧になり、テストもしづらい。malformed decision は corrective prompt を 1 回ないし 2 回まで再試行し、それでも壊れていれば安全に失敗させる。
  Date/Author: 2026-03-18 / Codex

## Outcomes & Retrospective

現時点では設計文書の作成までが完了しており、コードはまだ実装されていない。この文書により、次の実装者は「どのレイヤが何を判断するのか」「どこを新設し、どこは既存実装を再利用するのか」「どう検証すれば完了と言えるのか」を過去の会話なしで再現できる。未完了なのは実装、テスト、manual verification の三つである。

## Context and Orientation

この repository はドメインごとに package を分け、`data_model.py` に状態や request / result 型を置き、複雑な状態遷移は reducer pattern で実装する方針を取っている。副作用を持つ executor は reducer が返した command を実行し、テストは主に reducer の純粋ロジックに対して書く。

この作業で重要な既存ファイルは次のとおりである。`mini_rlm/repl_setup/setup.py` は REPL 環境を初期化し、ファイルや function collection を配置する。`mini_rlm/repl_session/executor.py` と `mini_rlm/repl_session/reducer.py` は一つの REPL session を進める。`mini_rlm/repl_session/run.py` の `execute_repl_session()` は setup, run, cleanup をまとめた高レベル API である。`mini_rlm/custom_functions/functions.py` は REPL 内の tool を定義し、その中の `rlm_query` が現在の再帰実行入口である。`mini_rlm/recursive_query/*` は child REPL を起動するための request / config / executor を持つ。`mini_rlm/system_prompt/system_prompt.py` は REPL session 用 system prompt を読み込む。`manual_tests/repl_pdf.py` と `manual_tests/repl_describe_image.py` は現状の REPL-centric manual verification である。

この文書で使う用語は次の意味に固定する。`Chat` はユーザーと上位エージェントの対話そのものを指す。`agent run` は Chat の途中で切り出される一回の delegated execution であり、self-contained な subtask を担当する。`REPL` は `agent run` の内部で必要に応じて使われる Python 実行環境であり、会話そのものではない。`planner model` は Chat レイヤで次の action を決める LLM 呼び出しを指す。`guardrail` は LLM の判断とは別にシステムが強制する deterministic な制約を指す。

今回のゴールは、現在の REPL-to-child-REPL recursion をそのまま Chat の概念に持ち込むことではない。むしろ Chat レイヤに新しい orchestrator を追加し、その orchestrator が「この turn はそのまま返答する」「この turn は `agent run` を起動する」のどちらかを明示的に選べるようにすることである。`agent run` 自体の中身には既存の `execute_repl_session()` を再利用してよいが、Chat 側は REPL 詳細に直接依存しない API を通して起動する。

## Plan of Work

最初のマイルストーンでは、Chat から見た中立的な `agent_run` package を追加する。これは新しい実行基盤を作るのではなく、既存の `execute_repl_session()` を「Chat が呼んでよい interface」として包み直す作業である。`mini_rlm/agent_run/data_model.py` に `AgentRunTemplate`, `AgentRunRequest`, `AgentRunResult` を定義し、`mini_rlm/agent_run/executor.py` に `execute_agent_run()` を置く。`AgentRunTemplate` は可変な prompt を除いた setup 情報を持ち、`ReplSetupRequest`, `ReplSessionLimits`, `RequestContext` を内包する。`AgentRunRequest` は template とその turn 固有の prompt を束ねる。`AgentRunResult` は `final_answer`, `termination_reason`, `total_tokens`, `model_token_usages`, `total_time_seconds` を持つ。実装は `execute_repl_session()` を呼び、その結果を `AgentRunResult` に写し替えるだけに留める。ここでは挙動を変えず、概念の境界を整えることが目的である。

同じマイルストーンの後半で、既存の `mini_rlm/recursive_query/executor.py` を `execute_agent_run()` 経由に切り替える。これにより、REPL 内の `rlm_query` と Chat レイヤの `agent run` が同じ高レベル実行 API を共有できる。recursive query 固有の責務である `remaining_depth` の減算、context 継承、file 継承、child limits の解決は引き続き `mini_rlm/recursive_query/*` に残し、`agent_run` package には移さない。ここでの狙いは「起動前の policy」は呼び出し側に残し、「起動後の実行」は一つの入口にまとめることにある。

二つ目のマイルストーンでは、新しい `mini_rlm/chat_session/` package を導入する。`data_model.py` には `ChatSessionRequest`, `ChatSessionLimits`, `ChatSessionResult`, `ChatPlannerAction`, `ChatPlannerDecision`, `ChatSessionState`, `ChatSessionCommand`, `ChatCommandResult` を置く。`ChatSessionRequest` は `conversation: list[MessageContent]`, planner 用 `request_context`, optional な `agent_run_template`, および `ChatSessionLimits` を受け取る。`ChatSessionLimits` は少なくとも `planner_iteration_limit`, `agent_run_limit`, `malformed_decision_limit` を持つ。`ChatSessionResult` は最終的にユーザーへ返す `assistant_message` と、何回 `agent run` を実行したかを返す。`ChatPlannerDecision` は `action` を `reply` または `agent_run` とし、`reason` を必須にする。`reply` の場合は `assistant_message` を、`agent_run` の場合は `agent_run_prompt` を必須にする。

`mini_rlm/chat_session/reducer.py` では、1 Chat turn を進める状態遷移を書く。初回は必ず `CALL_PLANNER` command を返す。planner が `reply` を返したら state に final assistant message を保存して `EXIT` に進む。planner が `agent_run` を返したら、まず guardrail を評価する。`agent_run_template` が無い場合、`agent_run_limit` を超えている場合、prompt が空文字の場合、直前と同じ `agent_run_prompt` を再度要求した場合は、その decision を拒否して corrective system message を conversation に追加し、再度 `CALL_PLANNER` に戻す。guardrail を通った場合だけ `EXECUTE_AGENT_RUN` command を返す。`agent run` 実行後は、child result をそのままユーザーに返さず、結果の要約を system message として conversation に追加してから再び `CALL_PLANNER` に戻す。これにより、最終的なユーザー向け回答は常に Chat レイヤが責任を持つ。

`mini_rlm/chat_session/executor.py` では reducer が返した command を dispatch する。`CALL_PLANNER` では `mini_rlm.llm.make_api_request()` を直接使い、会話履歴を `MessageContent` の role を保ったまま送る。ここで `text_query_with_usage()` は使わない。理由は、planner は複数 role の conversation を必要とし、単一 user message に flatten すると role 情報が失われるからである。planner 用 prompt は新規ファイル `mini_rlm/prompts/chat_session_prompt.txt` に置き、`mini_rlm/chat_session/prompt.py` に loader を作る。prompt では「簡単な質問はそのまま reply」「独立した探索が必要で、かつ一つの self-contained prompt に切り出せる場合だけ agent_run」という基準を明記する。レスポンスは厳密な JSON object 一つだけを返すよう要求する。executor はその JSON を parse し、失敗したら corrective system message を追加して再試行回数を増やす。

三つ目のマイルストーンでは、manual test と unit test を追加する。`manual_tests/chat_agent_run.py` を新設し、1 Chat turn を実行できる CLI を用意する。最低限 `--prompt`、`--toolset`、`--file`、`--model`、`--sub_model` を受け取るようにし、`toolset` は `minimal`, `image`, `pdf` から選ばせる。`minimal` はファイル不要で direct-reply path を見るために使い、`image` と `pdf` は agent-run path を見るために使う。`tests/chat_session/test_chat_session_reducer.py` では reducer の純粋ロジックを振る舞いベースでテストする。`tests/chat_session/test_chat_session_executor.py` では fake planner response と fake `execute_agent_run()` を使い、direct reply、single agent run、guardrail rejection、malformed JSON retry を検証する。必要なら `tests/agent_run/test_executor.py` を追加し、wrapper が `execute_repl_session()` を正しく使うことを押さえる。

## Milestone 1: Chat から見た `agent run` interface を作る

このマイルストーンの終わりには、「Chat が REPL package を直接知らなくても delegated execution を起動できる」という状態ができる。まだ Chat オーケストレータ自体は無いが、`agent run` という語で扱うに足る高レベル API が repository 内に存在し、recursive query もその API を共有する。

編集対象は `mini_rlm/agent_run/__init__.py`, `mini_rlm/agent_run/data_model.py`, `mini_rlm/agent_run/executor.py` の新規追加と、`mini_rlm/recursive_query/executor.py` の切り替えである。新しい package には class や helper を増やしすぎず、existing REPL path の薄い wrapper に徹する。acceptance は unit test で `AgentRunResult` が `execute_repl_session()` の observable fields を正しく引き継ぐこと、既存の recursive query テストが通ること、そして public import が `mini_rlm.agent_run` package root から可能なことの三点で確認する。

## Milestone 2: `chat_session` reducer / executor を追加する

このマイルストーンの終わりには、1 Chat turn を処理して、最終的に `assistant_message` を返す orchestrator が存在する。重要なのは、この orchestrator 自身が `agent run` を勝手に起動しないことである。planner LLM が明示的に `agent_run` action を返した場合だけ executor が起動し、それ以外は direct reply で終了する。

編集対象は `mini_rlm/chat_session/__init__.py`, `mini_rlm/chat_session/data_model.py`, `mini_rlm/chat_session/reducer.py`, `mini_rlm/chat_session/executor.py`, `mini_rlm/chat_session/run.py`, `mini_rlm/chat_session/prompt.py`, `mini_rlm/prompts/chat_session_prompt.txt` である。`run.py` には `execute_chat_session(request: ChatSessionRequest) -> ChatSessionResult` を置く。executor loop は `CALL_PLANNER -> EXECUTE_AGENT_RUN -> CALL_PLANNER -> EXIT` のような小さな state machine で十分であり、REPL session と同じ reducer pattern を採用する。acceptance は unit test で direct reply path と agent-run path の両方が観測できること、guardrail が deterministic に働くこと、malformed JSON が無限ループしないことによって確認する。

## Milestone 3: Manual verification と運用上の観測点を追加する

このマイルストーンの終わりには、開発者がローカル環境で「この prompt では `agent run` が起動しない」「この prompt では起動する」を目視確認できる。ここでの主役は `manual_tests/chat_agent_run.py` と、ログに出す少量の観測情報である。

manual script は `planner_action`, `agent_runs_executed`, `final_assistant_message`, 必要なら `last_agent_run_prompt` を標準出力に出す。これにより、ユーザーに見せる答えだけでなく、なぜその経路になったかも確認できる。acceptance は、簡単な質問で `agent_runs_executed=0` が出ることと、画像または PDF を含む質問で `agent_runs_executed=1` 以上が出ることにする。外部 API credentials が無い環境では、manual verification はスキップして unit test のみを通せばよいが、その場合もスキップ理由を記録する。

## Concrete Steps

最初に repository root で現状を確認する。

    cd /Users/takada-at/projects/mini_rlm
    rg -n "create_rlm_query|execute_recursive_query|execute_repl_session|make_api_request" mini_rlm

`agent_run` package を追加したら、relevant tests を先に回す。

    cd /Users/takada-at/projects/mini_rlm
    uv run pytest tests/recursive_query/test_executor.py -v --tb=short
    uv run pytest tests/repl_session/test_repl_session_reducer.py -v --tb=short

`chat_session` package と tests を追加したら、新規テストを個別に回す。

    cd /Users/takada-at/projects/mini_rlm
    uv run pytest tests/chat_session/test_chat_session_reducer.py -v --tb=short
    uv run pytest tests/chat_session/test_chat_session_executor.py -v --tb=short

全体検証は repository 標準のコマンドを使う。

    cd /Users/takada-at/projects/mini_rlm
    make format
    make lint
    make typecheck
    make test

成功時は `ruff` と `mypy` が 0 exit status を返し、`pytest` は新規テストを含めて全件 pass する。テスト件数は増えるので、固定値ではなく「既存件数 + 新規 chat_session / agent_run tests が pass」を確認する。

manual verification を行う場合は、既存 manual test と同じ環境変数を使う。少なくとも `MINI_RLM_LLM_ENDPOINT`, `MINI_RLM_LLM_API_KEY`, `MINI_RLM_LLM_MODEL` が必要である。direct reply path はファイル無しで試す。

    cd /Users/takada-at/projects/mini_rlm
    uv run python manual_tests/chat_agent_run.py --toolset minimal --prompt "2+2 を答えて"

期待する出力の要点は、`planner_action=reply` と `agent_runs_executed=0` が含まれること、そして最終返答が算術結果を含むことである。

agent-run path は既存の画像 fixture を使って試す。

    cd /Users/takada-at/projects/mini_rlm
    uv run python manual_tests/chat_agent_run.py --toolset image --file manual_tests/images/hello_world.png --prompt "画像の内容を説明して"

期待する出力の要点は、少なくとも一度 `planner_action=agent_run` が観測され、最終的に `agent_runs_executed=1` 以上になり、返答が画像の内容を説明していることである。

## Validation and Acceptance

受け入れ条件は、コードが存在することではなく、観測可能な振る舞いが得られることである。第一に、unit test で planner が明示的に `reply` を返したとき executor が `agent run` を起動しないことを示す。第二に、planner が `agent_run` を返したときだけ `execute_agent_run()` が呼ばれ、その結果が conversation に戻されてから最終 reply が生成されることを示す。第三に、同じ `agent_run_prompt` の連打、limit 超過、malformed decision が deterministic に拒否または停止されることを示す。

人間が目視で確認する acceptance は manual script で行う。簡単な質問で direct reply path に入ること、ファイルを伴う質問で agent-run path に入ること、そしてどちらの経路でも最終的にユーザー向け `assistant_message` が返ることを確認する。`agent run` の起動回数と最終メッセージを標準出力に出すことで、観測可能性を確保する。

## Idempotence and Recovery

この作業は additive に進める。`agent_run` package の追加、`chat_session` package の追加、prompt file の追加、test file の追加は何度繰り返しても destructive ではない。途中で reducer 実装に迷った場合は、新規 test file を先に書き、そこから state と command の最小集合を逆算する。manual test は外部 API に依存するため、credentials 不足で失敗した場合は unit test までで止めてよいが、その事実を `Progress` と `Outcomes & Retrospective` に明記する。

もし planner の JSON parser が不安定で進捗が止まった場合は、JSON 抽出 heuristic を増やすのではなく、planner prompt を強化し、malformed response に対する corrective system message を 1 回ずつ追加して再試行する。parser 自体は「厳密 JSON object を parse する」以外の責務を持たせない。この方針を崩す場合は、必ず `Decision Log` を更新する。

## Artifacts and Notes

planner の期待レスポンスは次のどちらか一つである。余計な文章は付けない。

    {
      "action": "reply",
      "reason": "The request can be answered directly from the conversation without delegated exploration.",
      "assistant_message": "2+2 is 4."
    }

または次の形である。

    {
      "action": "agent_run",
      "reason": "The request requires iterative inspection of a local file.",
      "agent_run_prompt": "The image file hello_world.png is already available in the working directory. Describe its contents."
    }

manual script の成功例は次のような短い出力を目指す。

    planner_action=reply
    agent_runs_executed=0
    assistant_message=2+2 is 4.

agent-run path の成功例は次のような形を目指す。

    planner_action=agent_run
    last_agent_run_prompt=The image file hello_world.png is already available in the working directory. Describe its contents.
    agent_runs_executed=1
    assistant_message=The image contains the words "Hello World".

## Interfaces and Dependencies

`mini_rlm/agent_run/data_model.py` には次の interface を定義する。

    class AgentRunTemplate(BaseModel):
        setup: ReplSetupRequest
        limits: ReplSessionLimits | None = None
        session_request_context: RequestContext | None = None

    class AgentRunRequest(BaseModel):
        prompt: str
        template: AgentRunTemplate

    class AgentRunResult(BaseModel):
        termination_reason: str
        final_answer: str | None
        total_iterations: int
        total_tokens: int
        model_token_usages: list[ModelTokenUsage] = Field(default_factory=list)
        total_time_seconds: float

`mini_rlm/agent_run/executor.py` には次の関数を定義する。

    def execute_agent_run(request: AgentRunRequest) -> AgentRunResult:
        ...

`mini_rlm/chat_session/data_model.py` には少なくとも次の interface を定義する。

    class ChatPlannerAction(StrEnum):
        REPLY = "reply"
        AGENT_RUN = "agent_run"

    class ChatPlannerDecision(BaseModel):
        action: ChatPlannerAction
        reason: str
        assistant_message: str | None = None
        agent_run_prompt: str | None = None

    class ChatSessionLimits(BaseModel):
        planner_iteration_limit: int = 4
        agent_run_limit: int = 2
        malformed_decision_limit: int = 2

    class ChatSessionRequest(BaseModel):
        request_context: RequestContext
        conversation: list[MessageContent]
        agent_run_template: AgentRunTemplate | None = None
        limits: ChatSessionLimits = Field(default_factory=ChatSessionLimits)

    class ChatSessionResult(BaseModel):
        assistant_message: str
        planner_iterations: int
        agent_runs_executed: int
        agent_run_results: list[AgentRunResult] = Field(default_factory=list)

`mini_rlm/chat_session/reducer.py` と `mini_rlm/chat_session/executor.py` は、既存の `repl_session` と同じ reducer pattern を採用する。最低限必要な command は `CALL_PLANNER`, `EXECUTE_AGENT_RUN`, `EXIT` である。planner 呼び出しには `mini_rlm.llm.make_api_request()` を使い、会話履歴は `list[MessageContent]` のまま渡す。direct reply の最終文面は planner が決めるが、guardrail の適用、malformed response の retry 回数、`agent run` の起動可否は reducer / executor 側で deterministic に決める。

Revision note (2026-03-18 / Codex): 初版を追加した。目的は、「Chat 中にいつ agent run を起動するか」という議論を、用語定義、設計判断、実装順序、検証方法まで含めて自己完結な ExecPlan に固定するためである。
