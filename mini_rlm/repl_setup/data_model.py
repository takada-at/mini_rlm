from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from mini_rlm.custom_functions import FunctionCollection
from mini_rlm.llm import RequestContext
from mini_rlm.recursive_query import RecursiveQueryRuntime
from mini_rlm.repl import ReplState


class ReplSetupRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_context: RequestContext
    setup_code: str | None = None
    context_payload: dict[str, Any] | list[Any] | str | None = None
    file_paths: list[Path] | None = None
    files: list["ReplFileRef"] | None = None
    functions: FunctionCollection | None = None
    recursive_query_runtime: RecursiveQueryRuntime | None = None


class ReplFileRef(BaseModel):
    source_path: Path
    target_name: str | None = None


class ReplContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_context: RequestContext
    repl_state: ReplState
    functions: FunctionCollection | None = None
    recursive_query_runtime: RecursiveQueryRuntime | None = None
