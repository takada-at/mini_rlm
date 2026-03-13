from typing import Any, Dict, Optional

from requests import Session

from mini_rlm.llm.data_model import Endpoint, RequestContext


def create_request_context(
    endpoint_url: str,
    model: str,
    api_key: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
) -> RequestContext:
    session = Session()
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    endpoint = Endpoint(url=endpoint_url, headers=headers)
    kwargs = {
        "model": model,
    }
    if request_params:
        kwargs.update(request_params)
    return RequestContext(session=session, endpoint=endpoint, kwargs=kwargs)
