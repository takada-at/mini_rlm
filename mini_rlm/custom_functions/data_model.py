from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel

from mini_rlm.llm.data_model import RequestContext


class Argument(BaseModel):
    name: str
    description: str
    type: Type


class FunctionBase(BaseModel):
    name: str
    description: str
    arguments: List[Argument]
    return_type: Optional[Type] = None


class FunctionFactory(FunctionBase):
    factory: Callable[[RequestContext], Callable[..., Any]]


class Function(FunctionBase):
    function: Callable[..., Any]


class FunctionCollection(BaseModel):
    functions: List[FunctionBase]
