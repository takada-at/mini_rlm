from threading import Lock
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ReplResult(BaseModel):
    stdout: str
    stderr: str
    locals: dict[str, Any]
    execution_time: float
    final_answer: str | None = None


class ReplState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    globals: Dict[str, Any] = Field(default_factory=dict)
    locals: Dict[str, Any] = Field(default_factory=dict)
    temp_dir: str = ""
    lock: Lock = Field(default_factory=Lock)
    last_final_answer: str | None = None
    context_count: int = 0
    history_count: int = 0
    # scaffold functions to restore after each execution
    reserved_globals: dict[str, Any] = Field(default_factory=dict)
