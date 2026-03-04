import base64
from io import BytesIO

from PIL import Image

from mini_rlm.image.data_model import ImageData


def open_image_data(file_path: str) -> ImageData:
    """Open an image file and convert it to ImageData."""
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    mime_type = (
        Image.open(BytesIO(image_bytes)).get_format_mimetype()
        or "application/octet-stream"
    )
    return ImageData(bytes=image_bytes, mime_type=mime_type)


def convert_pil_image_to_image_data(
    image: Image.Image, format: str = "PNG"
) -> ImageData:
    """Convert a PIL Image to ImageData."""
    with BytesIO() as output:
        image.save(output, format=format)
        image_bytes = output.getvalue()
    mime_type = f"image/{format.lower()}"
    return ImageData(bytes=image_bytes, mime_type=mime_type)


def convert_image_data_to_pil_image(image_data: ImageData) -> Image.Image:
    """Convert ImageData to a PIL Image."""
    return Image.open(BytesIO(image_data.bytes))


def convert_image_data_to_image_url(image_data: ImageData) -> str:
    """Convert ImageData to a data URL."""
    encoded_bytes = base64.b64encode(image_data.bytes)
    encoded_str = encoded_bytes.decode("utf-8")
    return f"data:{image_data.mime_type};base64,{encoded_str}"
