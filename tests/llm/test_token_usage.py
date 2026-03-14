from mini_rlm.llm.data_model import APIRequestResult, ModelTokenUsage
from mini_rlm.llm.token_usage import (
    get_detailed_token_usage_from_response,
    merge_model_token_usages,
)


def test_get_detailed_token_usage_from_response_extracts_model_scoped_usage() -> None:
    # give: OpenAI Compatible な usage を含む response がある
    response = APIRequestResult(
        response_json={
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        },
        messages=[],
        resolved_model_name="gpt-test",
    )

    # when: 詳細 usage を抽出する
    token_usage = get_detailed_token_usage_from_response(response)

    # then: total とモデル別 usage が両方取り出せる
    assert token_usage.total_tokens == 18
    assert token_usage.model_token_usages == [
        ModelTokenUsage(
            model_name="gpt-test",
            prompt_tokens=11.0,
            completion_tokens=7.0,
        )
    ]


def test_merge_model_token_usages_merges_same_and_different_models() -> None:
    # give: 既存 usage と追加 usage に同一モデルと別モデルが混在している
    existing = [
        ModelTokenUsage(
            model_name="gpt-4.1",
            prompt_tokens=10.0,
            completion_tokens=5.0,
        )
    ]
    new = [
        ModelTokenUsage(
            model_name="gpt-4.1",
            prompt_tokens=3.0,
            completion_tokens=2.0,
        ),
        ModelTokenUsage(
            model_name="gpt-4.1-mini",
            prompt_tokens=7.0,
            completion_tokens=1.0,
        ),
    ]

    # when: モデル別 usage をマージする
    merged = merge_model_token_usages(existing, new)

    # then: 同一モデルは加算され、別モデルは保持される
    assert merged == [
        ModelTokenUsage(
            model_name="gpt-4.1",
            prompt_tokens=13.0,
            completion_tokens=7.0,
        ),
        ModelTokenUsage(
            model_name="gpt-4.1-mini",
            prompt_tokens=7.0,
            completion_tokens=1.0,
        ),
    ]
