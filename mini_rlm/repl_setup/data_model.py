from pydantic import BaseModel, ConfigDict

from mini_rlm.custom_functions.data_model import FunctionCollection
from mini_rlm.llm.data_model import RequestContext
from mini_rlm.repl.repl import ReplState


class ReplContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_context: RequestContext
    repl_state: ReplState
    functions: FunctionCollection | None = None
