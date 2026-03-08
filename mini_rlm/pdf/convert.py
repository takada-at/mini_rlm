import os

import pypdfium2 as pdfium

from mini_rlm.image.convert import convert_pil_image_to_image_data
from mini_rlm.image.data_model import ImageData


def convert_pdf_page_to_image_data(file_path: str, page_index: int) -> ImageData:
    """Convert a PDF page to ImageData."""
    page = _get_page(file_path, page_index)
    pil_image = page.render().to_pil()
    return convert_pil_image_to_image_data(pil_image)


def convert_pdf_page_to_text(file_path: str, page_index: int) -> str:
    """Extract text from a PDF page."""
    page = _get_page(file_path, page_index)
    return page.get_textpage().get_text_range()


def _get_page(file_path: str, page_index: int) -> pdfium.PdfPage:
    """Helper function to get a PDF page object."""
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Invalid PDF file: {file_path}")
    pdf = pdfium.PdfDocument(file_path)
    if page_index < 0 or page_index >= len(pdf):
        raise ValueError(
            f"Page number {page_index} is out of range for PDF with {len(pdf)} pages."
        )
    return pdf.get_page(page_index)
