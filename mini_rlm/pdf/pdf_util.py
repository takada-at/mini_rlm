import pypdfium2 as pdfium


def pdf_page_length(pdf_path: str) -> int:
    """Return the number of pages in a PDF file."""
    pdf = pdfium.PdfDocument(pdf_path)
    return len(pdf)
