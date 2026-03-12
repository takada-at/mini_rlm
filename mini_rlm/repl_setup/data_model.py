from pydantic import BaseModel, ConfigDict

from mini_rlm.custom_functions import FunctionCollection
from mini_rlm.llm import RequestContext
from mini_rlm.recursive_query import RecursiveQueryRuntime
from mini_rlm.repl import ReplState


class ReplContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_context: RequestContext
    repl_state: ReplState
    functions: FunctionCollection | None = None
    recursive_query_runtime: RecursiveQueryRuntime | None = None
