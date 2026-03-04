from typing import Dict, List

from pydantic import BaseModel
from requests import Session


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class MessageContentPart(BaseModel):
    type: str  # "text" or "image"
    text: str | None = None
    image_url: ImageURL | None = None


class MessageContent(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: List[MessageContentPart] | str
    name: str | None = None


class Endpoint(BaseModel):
    url: str
    method: str = "GET"
    headers: Dict[str, str] | None = None
    body: str | None = None


class RequestContext(BaseModel):
    session: Session
    endpoint: Endpoint
    kwargs: Dict[str, str] | None = None
    messages: List[MessageContent] | None = None
