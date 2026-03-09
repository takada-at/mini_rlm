from enum import StrEnum
from threading import Lock
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class ReplResult(BaseModel):
    stdout: str
    stderr: str
    locals: dict[str, Any]
    execution_time: float
    final_answer: str | None = None
    expression_result: str | None = None


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


class ReplExecutionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ReplCommandType(StrEnum):
    EXECUTE_STATEMENTS = "execute_statements"
    EVALUATE_EXPRESSION = "evaluate_expression"
    COMPLETE = "complete"
    EXIT = "exit"


class ReplCommand(BaseModel):
    type: ReplCommandType
    code: str | None = None


class ReplCommandResultType(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ReplCommandResult(BaseModel):
    command_type: ReplCommandType
    type: ReplCommandResultType
    stdout: str = ""
    stderr: str = ""
    expression_result: str | None = None


class ReplExecutionState(BaseModel):
    code: str
    status: ReplExecutionStatus
    statement_code: str = ""
    final_expression_code: str | None = None
    stdout: str = ""
    stderr: str = ""
    expression_result: str | None = None
    last_command_type: ReplCommandType | None = None
