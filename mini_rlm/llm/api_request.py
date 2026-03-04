from typing import Any, Dict, List

from mini_rlm.llm.data_model import MessageContent, RequestContext


def dump_messages(messages: List[MessageContent]) -> List[Dict[str, Any]]:
    """Dump a list of MessageContent to a list of dicts for JSON serialization."""
    return [message.model_dump() for message in messages]


def make_api_request(
    context: RequestContext, messages: List[MessageContent]
) -> List[MessageContent]:
    """Make an API request to the endpoint specified in *context* with the given *messages*."""
    dict_messages = dump_messages(messages)
    if context.messages:
        dict_messages = dump_messages(context.messages) + dict_messages
    request_body: Dict[str, Any] = {"messages": dict_messages}
    if context.kwargs:
        request_body.update(context.kwargs)

    response = context.session.request(
        "POST",
        context.endpoint.url,
        headers=context.endpoint.headers,
        json=request_body,
    )

    response.raise_for_status()
    # For simplicity, we assume the response is a JSON array of MessageContent
    response_data = response.json()
    if (
        "choices" in response_data
        and isinstance(response_data["choices"], list)
        and len(response_data["choices"]) > 0
    ):
        return [
            MessageContent.model_validate(choice["message"])
            for choice in response_data["choices"]
        ]
    else:
        return []
