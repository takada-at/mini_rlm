from dataclasses import field
from threading import Lock
from typing import Any

from pydantic import BaseModel, ConfigDict


class ReplResult(BaseModel):
    stdout: str
    stderr: str
    locals: dict[str, Any]
    execution_time: float
    final_answer: str | None = None


class ReplState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    temp_dir: str = ""
    lock: Lock = field(default_factory=Lock)
    last_final_answer: str | None = None
    context_count: int = 0
    history_count: int = 0
    # scaffold functions to restore after each execution
    reserved_globals: dict[str, Any] = field(default_factory=dict)
