from mini_rlm.llm import (
    ImageURL,
    MessageContent,
    MessageContentPart,
    ResponseItem,
    create_request_context,
)
from mini_rlm.llm.data_model import APIType
from mini_rlm.repl_session.convert import (
    estimate_session_history_tokens,
    resolve_session_limits,
)
from mini_rlm.repl_session.data_model import (
    ReplSessionLimits,
    ReplSessionState,
    ReplSessionStatus,
)


def test_prefix_and_opaque_items_count_without_duplicating_compacted_prompt() -> None:
    # テストしたいふるまい: システム指示と推論を含め、圧縮に含まれる質問は重複計上しない
    # give: prefixと質問を保持した圧縮結果
    state = ReplSessionState(
        prompt="question" * 100,
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=10000, iteration_limit=10, timeout_seconds=60, error_threshold=5
        ),
        started_at_seconds=0,
        current_time_seconds=0,
        input_prefix=[
            MessageContent(role="system", content="instructions"),
            MessageContent(role="user", content="data" * 100),
        ],
        messages=[
            ResponseItem.model_validate(
                {"type": "compaction", "encrypted_content": "opaque" * 100}
            )
        ],
        history_includes_prompt=True,
    )
    # when: 圧縮前扱いと圧縮後扱いを比較する
    compacted = estimate_session_history_tokens(state)
    duplicated = estimate_session_history_tokens(
        state.model_copy(update={"history_includes_prompt": False})
    )
    # then: opaque内容は非ゼロ、圧縮後にprefix userと質問は再加算されない
    assert compacted > 600
    assert duplicated > compacted + 1200
    assert (
        estimate_session_history_tokens(state.model_copy(update={"input_prefix": []}))
        < compacted
    )


def test_image_allowance_is_independent_of_transport_encoding() -> None:
    # テストしたいふるまい: 画像のURLやbase64の文字数を画像トークン数として数えない
    # give: 同じ画像予算と異なる転送表現
    state = ReplSessionState(
        prompt="describe",
        status=ReplSessionStatus.RUNNING,
        limits=ReplSessionLimits(
            token_limit=10000,
            iteration_limit=10,
            timeout_seconds=60,
            error_threshold=5,
            image_token_estimate=5000,
        ),
        started_at_seconds=0,
        current_time_seconds=0,
    )
    # when: URL画像と長いbase64画像を見積もる
    sizes = [
        estimate_session_history_tokens(
            state.model_copy(
                update={
                    "messages": [
                        MessageContent(
                            role="user",
                            content=[
                                MessageContentPart(
                                    type="image_url", image_url=ImageURL(url=url)
                                )
                            ],
                        )
                    ]
                }
            )
        )
        for url in (
            "https://example.invalid/image.png",
            "data:image/png;base64," + "A" * 100000,
        )
    ]
    # then: 両方に同じ設定値を割り当てる
    assert sizes[0] == sizes[1]
    assert sizes[0] >= 5000


def test_generation_limit_increases_reserve_without_changing_budget() -> None:
    # テストしたいふるまい: APIの生成上限を出力余裕に反映し、入力設定を変更しない
    # give: 独立した消費予算と小さい出力余裕
    limits = ReplSessionLimits(
        token_limit=999999,
        iteration_limit=10,
        timeout_seconds=60,
        error_threshold=5,
        output_token_reserve=100,
    )
    cases: list[tuple[APIType, str]] = [
        ("responses", "max_output_tokens"),
        ("chat_completions", "max_completion_tokens"),
        ("chat_completions", "max_tokens"),
    ]
    for api_type, key in cases:
        context = create_request_context(
            "https://example.invalid/v1/responses", "test", api_type=api_type
        )
        context.kwargs = {key: 5000}
        # when: 実際の生成設定を反映する
        resolved = resolve_session_limits(limits, context)
        # then: 出力余裕だけが増え、元の設定と消費予算は変わらない
        assert resolved.output_token_reserve == 5000
        assert resolved.token_limit == limits.token_limit
        assert limits.output_token_reserve == 100
        assert context.kwargs == {key: 5000}
