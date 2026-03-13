from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel, ConfigDict

from mini_rlm.llm import RequestContext
from mini_rlm.recursive_query import RecursiveQueryRuntime
from mini_rlm.repl import ReplState


class Argument(BaseModel):
    name: str
    description: str
    type: Type


class FunctionBase(BaseModel):
    name: str
    description: str
    arguments: List[Argument]
    return_type: Optional[Type] = None


class FunctionCollection(BaseModel):
    functions: List[FunctionBase]


class FunctionFactoryContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_context: RequestContext
    repl_state: ReplState
    function_collection: FunctionCollection | None = None
    recursive_query_runtime: RecursiveQueryRuntime | None = None


class FunctionFactory(FunctionBase):
    factory: Callable[[FunctionFactoryContext], Callable[..., Any]]


class Function(FunctionBase):
    function: Callable[..., Any]
