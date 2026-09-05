"""Pure conversion between domain messages and the two supported wire formats."""

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from mini_rlm.llm.data_model import (
    APIType,
    HistoryItem,
    MessageContent,
    ParsedResponse,
    RequestContext,
    RequestOperation,
    RequestPayload,
    RequestResultType,
    ResponseItem,
)


def dump_messages(messages: Sequence[HistoryItem]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        if not isinstance(message, MessageContent):
            raise ValueError("Responses items cannot be sent to Chat Completions.")
        result.append(message.model_dump(exclude_none=True))
    return result


def dump_response_input(messages: Sequence[HistoryItem]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        if isinstance(message, ResponseItem):
            result.append(message.model_dump(exclude_unset=True))
            continue
        if isinstance(message.content, str):
            content = [{"type": "input_text", "text": message.content}]
        else:
            content = []
            for part in message.content:
                if part.type == "text" and part.text is not None:
                    content.append({"type": "input_text", "text": part.text})
                elif part.type == "image_url" and part.image_url is not None:
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": part.image_url.url,
                            "detail": part.image_url.detail,
                        }
                    )
                else:
                    raise ValueError("Message content part is missing its value.")
        result.append({"role": message.role, "content": content})
    return result


def build_request_payload(
    context: RequestContext,
    messages: Sequence[HistoryItem],
    *,
    operation: RequestOperation = "create",
    include_context_messages: bool = True,
) -> RequestPayload:
    history = list(context.messages or []) if include_context_messages else []
    all_messages: list[HistoryItem] = [*history, *messages]
    params = dict(context.kwargs or {})
    url = context.endpoint.url
    if context.api_type == "chat_completions":
        if operation != "create":
            raise ValueError("Compaction requires the Responses API.")
        body = {"messages": dump_messages(all_messages), **params}
    else:
        for key in ("previous_response_id", "conversation", "messages", "input"):
            if key in params:
                raise ValueError(
                    f"{key} conflicts with locally managed Responses history."
                )
        if params.get("stream") or params.get("background"):
            raise ValueError("Streaming and background responses are not supported.")
        response_input = dump_response_input(all_messages)
        if operation == "compact":
            # Generation-only settings (store, reasoning, include, etc.) are not
            # accepted by /responses/compact.
            body = {
                key: value
                for key, value in params.items()
                if key in ("model", "instructions", "prompt_cache_key", "service_tier")
            }
            body["input"] = response_input
            parts = urlsplit(url)
            url = urlunsplit(parts._replace(path=parts.path.rstrip("/") + "/compact"))
        else:
            include = params.get("include") or []
            if not isinstance(include, list) or not all(
                isinstance(v, str) for v in include
            ):
                raise ValueError("include must be a list of strings.")
            body = {
                **params,
                "input": response_input,
                "store": False,
                "include": list(
                    dict.fromkeys([*include, "reasoning.encrypted_content"])
                ),
            }
    return RequestPayload(
        url=url,
        headers=context.endpoint.headers or {},
        body=body,
        timeout_seconds=120.0,
        api_type=context.api_type,
        operation=operation,
    )


def parse_response(
    response: dict[str, Any],
    api_type: APIType,
    operation: RequestOperation = "create",
) -> ParsedResponse:
    if not isinstance(response, dict):
        raise ValueError("Response JSON must be an object.")
    if api_type == "chat_completions":
        choices = response.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise ValueError("response JSON does not contain valid 'choices'")
        return ParsedResponse(
            messages=[MessageContent.model_validate(choices[0].get("message"))]
        )

    if operation == "compact":
        if response.get("object") != "response.compaction":
            raise ValueError("Expected a response.compaction object.")
    elif response.get("status") != "completed":
        raise ValueError(
            f"Response did not complete: {response.get('status')}; {response.get('error') or response.get('incomplete_details')}"
        )
    output = response.get("output")
    if not isinstance(output, list) or not output:
        raise ValueError("Response does not contain non-empty output.")
    items = [ResponseItem.model_validate(item) for item in output]
    if operation == "compact":
        if not any(
            item.type == "compaction" and item.model_dump().get("encrypted_content")
            for item in items
        ):
            raise ValueError(
                "Compaction response is missing encrypted compaction content."
            )
        return ParsedResponse(output_items=items)

    messages = []
    for item in output:
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        if item.get("status", "completed") != "completed":
            raise ValueError("Assistant message did not complete.")
        content = item.get("content")
        if not isinstance(content, list):
            raise ValueError("Assistant message content must be a list.")
        texts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid assistant content part.")
            if part.get("type") == "output_text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("output_text must contain text.")
                texts.append(part["text"])
        if texts:
            messages.append(MessageContent(role="assistant", content="".join(texts)))
    return ParsedResponse(messages=messages, output_items=items)


def classify_response_error(
    response: Any, api_type: APIType, operation: RequestOperation
) -> RequestResultType:
    """Separate generation failures from malformed responses that may be retried."""
    if api_type == "responses" and operation == "create" and isinstance(response, dict):
        if response.get("status") in ("incomplete", "failed", "cancelled"):
            return RequestResultType.INCOMPLETE_RESPONSE
    return RequestResultType.INVALID_RESPONSE
