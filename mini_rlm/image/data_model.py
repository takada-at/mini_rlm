from pydantic import BaseModel


class ImageData(BaseModel):
    bytes: bytes
    mime_type: str
