from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel, ConfigDict

from mini_rlm.llm.data_model import RequestContext
from mini_rlm.repl.data_model import ReplState


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


class FunctionFactory(FunctionBase):
    factory: Callable[[FunctionFactoryContext], Callable[..., Any]]


class Function(FunctionBase):
    function: Callable[..., Any]
