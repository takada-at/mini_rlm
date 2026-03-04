from typing import Any, Dict

from requests import Session

from mini_rlm.llm.data_model import Endpoint, RequestContext


def create_request_context(
    endpoint: Endpoint, request_params: Dict[str, Any]
) -> RequestContext:
    session = Session()
    return RequestContext(session=session, endpoint=endpoint, kwargs=request_params)
