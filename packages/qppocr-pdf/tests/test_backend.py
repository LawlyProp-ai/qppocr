"""Backend signal / rasterization / text tests against synthetic PDFs."""

import io

import pypdfium2 as pdfium
from PIL import Image

from qppocr_pdf import backend, extract_text, page_signals, render_page_png


def _page(data: bytes):
    doc = pdfium.PdfDocument(data)
    return doc, doc.get_page(0)


def test_text_page_has_text_and_no_image_coverage(text_pdf):
    doc, page = _page(text_pdf)
    has_text, coverage = page_signals(doc, page)
    assert has_text is True
    assert coverage == 0.0
    doc.close()


def test_scan_page_has_no_text_and_full_coverage(scan_pdf):
    doc, page = _page(scan_pdf)
    has_text, coverage = page_signals(doc, page)
    assert has_text is False
    assert coverage == 1.0
    doc.close()


def test_logo_coverage_is_small(logo_pdf):
    doc, page = _page(logo_pdf)
    has_text, coverage = page_signals(doc, page)
    assert has_text is True
    assert 0.0 < coverage < 0.5
    doc.close()


def test_extract_text_returns_known_string(text_pdf):
    doc, page = _page(text_pdf)
    assert "Hello PDF World" in extract_text(page)
    doc.close()


def test_extract_text_normalizes_crlf(text_pdf, monkeypatch):
    doc, page = _page(text_pdf)
    textpage = page.get_textpage()
    monkeypatch.setattr(textpage, "get_text_bounded", lambda *a, **k: "a\r\nb\rc")
    monkeypatch.setattr(page, "get_textpage", lambda: textpage)
    assert extract_text(page) == "a\nb\nc"
    doc.close()


def test_extract_text_empty_on_scan(scan_pdf):
    doc, page = _page(scan_pdf)
    assert extract_text(page) == ""
    doc.close()


def test_render_page_png_dimensions_and_mode(text_pdf):
    doc, page = _page(text_pdf)
    width_pt, height_pt = page.get_size()
    dpi = 144
    png = render_page_png(page, dpi)
    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG"
    assert image.mode == "RGB"
    assert image.size == (round(width_pt * dpi / 72), round(height_pt * dpi / 72))
    doc.close()


def test_deeply_nested_objects_terminate_cleanly(deep_nested_pdf):
    # The image is nested deeper than MAX_OBJECT_DEPTH: traversal must return
    # (no RecursionError, no hang) whether or not the image is detected.
    doc, page = _page(deep_nested_pdf)
    has_text, coverage = page_signals(doc, page)
    assert isinstance(has_text, bool)
    assert 0.0 <= coverage <= 1.0
    doc.close()


def test_get_objects_depth_is_bounded():
    captured = {}

    class _TextPage:
        def count_chars(self):
            return 0

    class _SpyPage:
        def get_textpage(self):
            return _TextPage()

        def get_bbox(self):
            return (0.0, 0.0, 612.0, 792.0)

        def get_objects(self, filter=None, max_depth=None):
            captured["filter"] = filter
            captured["max_depth"] = max_depth
            return []

    page_signals(None, _SpyPage())
    assert captured["max_depth"] == backend.MAX_OBJECT_DEPTH
    assert captured["filter"] == [backend._IMAGE]
