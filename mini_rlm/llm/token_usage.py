from mini_rlm.llm.data_model import APIRequestResult, ModelTokenUsage, TokenUsage


def get_token_usage_from_response(response: APIRequestResult) -> int:
    """Extract the total token usage from the LLM response JSON."""
    return get_detailed_token_usage_from_response(response).total_tokens


def get_detailed_token_usage_from_response(response: APIRequestResult) -> TokenUsage:
    """Extract total and model-scoped token usage from the LLM response JSON."""
    response_json = response.response_json
    if "usage" in response_json and isinstance(response_json["usage"], dict):
        usage = response_json["usage"]
        total_tokens = usage.get("total_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        model_name = response.resolved_model_name

        token_usage = TokenUsage()
        if isinstance(total_tokens, int):
            token_usage.total_tokens = total_tokens

        if (
            isinstance(model_name, str)
            and model_name != ""
            and isinstance(prompt_tokens, (int, float))
            and isinstance(completion_tokens, (int, float))
        ):
            token_usage.model_token_usages = [
                ModelTokenUsage(
                    model_name=model_name,
                    prompt_tokens=float(prompt_tokens),
                    completion_tokens=float(completion_tokens),
                )
            ]
        return token_usage
    return TokenUsage()


def merge_model_token_usages(
    existing_usages: list[ModelTokenUsage],
    new_usages: list[ModelTokenUsage],
) -> list[ModelTokenUsage]:
    """Merge model-scoped token usages and return them sorted by model_name."""
    merged: dict[str, ModelTokenUsage] = {}
    for usage in existing_usages + new_usages:
        current = merged.get(usage.model_name)
        if current is None:
            merged[usage.model_name] = usage.model_copy(deep=True)
            continue
        merged[usage.model_name] = ModelTokenUsage(
            model_name=usage.model_name,
            prompt_tokens=current.prompt_tokens + usage.prompt_tokens,
            completion_tokens=current.completion_tokens + usage.completion_tokens,
        )
    return [merged[key] for key in sorted(merged)]


def diff_model_token_usages(
    previous_usages: list[ModelTokenUsage],
    current_usages: list[ModelTokenUsage],
) -> list[ModelTokenUsage]:
    """Return the per-model delta between two cumulative usage snapshots."""
    merged_previous = {
        usage.model_name: usage
        for usage in merge_model_token_usages([], previous_usages)
    }
    deltas: list[ModelTokenUsage] = []
    for usage in merge_model_token_usages([], current_usages):
        previous = merged_previous.get(usage.model_name)
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        if previous is not None:
            prompt_tokens -= previous.prompt_tokens
            completion_tokens -= previous.completion_tokens
        if prompt_tokens == 0 and completion_tokens == 0:
            continue
        deltas.append(
            ModelTokenUsage(
                model_name=usage.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
    return deltas
