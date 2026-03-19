from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class RunMode(StrEnum):
    AUTO = "auto"
    MINIMAL = "minimal"
    IMAGE = "image"
    PDF = "pdf"


class ChatCLIInputType(StrEnum):
    EMPTY = "empty"
    EXIT = "exit"
    HELP = "help"
    FILES = "files"
    ADD_FILE = "add_file"
    RESET = "reset"
    SEND_MESSAGE = "send_message"
    INVALID = "invalid"


class CommonCLIConfig(BaseModel):
    endpoint_url: str
    api_key: str
    model: str
    sub_model: str
    files: list[Path] = Field(default_factory=list)
    verbose: bool = False


class ChatCLIConfig(CommonCLIConfig):
    initial_prompt: str | None = None


class ChatCLIInput(BaseModel):
    type: ChatCLIInputType
    message: str | None = None
    file_path: Path | None = None
    force_run: bool = False
    error_message: str | None = None


class RunCLIConfig(CommonCLIConfig):
    prompt: str
    mode: RunMode = RunMode.AUTO
