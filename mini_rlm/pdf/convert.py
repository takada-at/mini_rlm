import os

import pypdfium2 as pdfium

from mini_rlm.image.convert import convert_pil_image_to_image_data
from mini_rlm.image.data_model import ImageData


def convert_pdf_to_image_data(file_path: str, index: int) -> ImageData:
    """Convert a PDF page to ImageData."""
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid PDF file: {file_path}")
    pdf = pdfium.PdfDocument(file_path)
    if index < 0 or index >= len(pdf):
        raise ValueError(
            f"Page number {index} is out of range for PDF with {len(pdf)} pages."
        )
    page = pdf.get_page(index)
    pil_image = page.render().to_pil()
    return convert_pil_image_to_image_data(pil_image)
