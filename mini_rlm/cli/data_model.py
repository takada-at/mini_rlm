from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class RunMode(StrEnum):
    AUTO = "auto"
    MINIMAL = "minimal"
    IMAGE = "image"
    PDF = "pdf"


class CommonCLIConfig(BaseModel):
    endpoint_url: str
    api_key: str
    model: str
    files: list[Path] = Field(default_factory=list)
    verbose: bool = False


class ChatCLIConfig(CommonCLIConfig):
    initial_prompt: str | None = None


class RunCLIConfig(CommonCLIConfig):
    prompt: str
    mode: RunMode = RunMode.AUTO
