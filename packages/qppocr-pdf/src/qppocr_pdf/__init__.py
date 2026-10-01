"""Per-page PDF routing between the embedded text layer and OCR.

The public entry point is :func:`qppocr_pdf.ocr_pdf`; classification is a pure
function (:func:`qppocr_pdf.decide`) so it can be tested without PDFs.
"""

from .backend import extract_text, page_signals, render_page_png
from .classify import decide
from .extract import (
    Char,
    EncryptedPdfError,
    Line,
    MalformedPdfError,
    PageResult,
    PageTooLargeError,
    PdfError,
    ocr_pdf,
)

__all__ = [
    "Char",
    "EncryptedPdfError",
    "Line",
    "MalformedPdfError",
    "PageResult",
    "PageTooLargeError",
    "PdfError",
    "decide",
    "extract_text",
    "ocr_pdf",
    "page_signals",
    "render_page_png",
]
