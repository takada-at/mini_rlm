from mini_rlm.llm.data_model import APIRequestResult


def get_token_usage_from_response(response: APIRequestResult) -> int:
    """Extract the total token usage from the LLM response JSON."""
    response_json = response.response_json
    if "usage" in response_json and isinstance(response_json["usage"], dict):
        usage = response_json["usage"]
        if "total_tokens" in usage and isinstance(usage["total_tokens"], int):
            return usage["total_tokens"]
    return 0
