from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel


class Argument(BaseModel):
    name: str
    description: str
    type: Type


class Function(BaseModel):
    name: str
    description: str
    arguments: List[Argument]
    function: Callable[..., Any]
    return_type: Optional[Type] = None


class FunctionCollection(BaseModel):
    functions: List[Function]
