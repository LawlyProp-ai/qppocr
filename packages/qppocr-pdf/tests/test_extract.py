"""Routing, geometry, force override and failure tests for ``ocr_pdf``."""

import pytest

from qppocr_pdf import (
    EncryptedPdfError,
    MalformedPdfError,
    PageTooLargeError,
    PdfError,
    ocr_pdf,
)


class _FakeChar:
    def __init__(self, text, pts):
        self.text = text
        self.pts = pts


class _FakeLine:
    def __init__(self, text, confidence, pts, chars=()):
        self.text = text
        self.confidence = confidence
        self.pts = pts
        self.chars = chars


class _FakeResult:
    def __init__(self, lines):
        self.lines = lines


#: Fixed line quad in *pixel* coordinates (fits a letter page at dpi=100).
_LINE_PTS = [(85.0, 110.0), (680.0, 110.0), (680.0, 220.0), (85.0, 220.0)]


class _FakeEngine:
    """Records calls and returns one fixed line in *pixel* coordinates."""

    def __init__(self):
        self.calls = 0
        self.images = []

    def run(self, image):
        assert isinstance(image, (bytes, bytearray))
        self.calls += 1
        self.images.append(image)
        char = _FakeChar("A", _LINE_PTS)
        line = _FakeLine("hello", 0.99, _LINE_PTS, chars=(char,))
        return _FakeResult([line])


@pytest.fixture
def engine():
    return _FakeEngine()


def test_text_only_routes_to_text_without_ocr(text_pdf, engine):
    result = ocr_pdf(text_pdf, engine)
    assert len(result) == 1
    page = result[0]
    assert page.kind == "text"
    assert "Hello PDF World" in page.text
    assert page.lines == []
    assert page.scale == 1.0
    assert engine.calls == 0


def test_scan_routes_to_ocr(scan_pdf, engine):
    result = ocr_pdf(scan_pdf, engine)
    assert result[0].kind == "ocr"
    assert engine.calls == 1
    assert result[0].lines
    assert result[0].text == "hello"


def test_mixed_above_threshold_routes_to_ocr(mixed_pdf, engine):
    result = ocr_pdf(mixed_pdf, engine)
    assert result[0].kind == "ocr"
    assert engine.calls == 1


def test_logo_below_threshold_stays_text(logo_pdf, engine):
    result = ocr_pdf(logo_pdf, engine)
    assert result[0].kind == "text"
    assert engine.calls == 0


def test_force_text_skips_ocr(scan_pdf, engine):
    result = ocr_pdf(scan_pdf, engine, force="text")
    assert result[0].kind == "text"
    assert engine.calls == 0


def test_force_ocr_ignores_text_layer(text_pdf, engine):
    result = ocr_pdf(text_pdf, engine, force="ocr")
    assert result[0].kind == "ocr"
    assert engine.calls == 1


def test_geometry_scaled_to_pdf_points(scan_pdf, engine):
    dpi = 100
    result = ocr_pdf(scan_pdf, engine, dpi=dpi)
    page = result[0]
    assert page.scale == pytest.approx(72.0 / dpi)
    assert page.dpi == dpi
    line = page.lines[0]
    for (px, py), (sx, sy) in zip(_LINE_PTS, line.pts):
        assert sx == pytest.approx(px * 72.0 / dpi)
        assert sy == pytest.approx(py * 72.0 / dpi)
        assert 0.0 <= sx <= page.width_pt
        assert 0.0 <= sy <= page.height_pt
    assert line.chars[0].text == "A"
    assert line.chars[0].pts[0] == pytest.approx(
        (_LINE_PTS[0][0] * 72.0 / dpi, _LINE_PTS[0][1] * 72.0 / dpi)
    )


def test_oversized_page_fails_closed(oversized_pdf, engine):
    with pytest.raises(PageTooLargeError):
        ocr_pdf(oversized_pdf, engine)
    assert engine.calls == 0


def test_pixel_cap_override(text_pdf, engine):
    with pytest.raises(PageTooLargeError):
        ocr_pdf(text_pdf, engine, force="ocr", max_pixels=100)


def test_pixel_cap_reflects_dpi(text_pdf, engine):
    # 612x792 pt at 200 DPI renders to 1700x2200 px (~3.74M).
    with pytest.raises(PageTooLargeError):
        ocr_pdf(text_pdf, engine, force="ocr", max_pixels=3_000_000)
    result = ocr_pdf(text_pdf, engine, force="ocr", max_pixels=4_000_000)
    assert result[0].kind == "ocr"


def test_malformed_pdf_raises(engine):
    with pytest.raises(MalformedPdfError):
        ocr_pdf(b"this is definitely not a pdf", engine)


def test_encrypted_pdf_raises(encrypted_pdf, engine):
    with pytest.raises(EncryptedPdfError):
        ocr_pdf(encrypted_pdf, engine)


def test_invalid_parameters(engine):
    with pytest.raises(ValueError):
        ocr_pdf(b"x", engine, force="sometimes")
    with pytest.raises(ValueError):
        ocr_pdf(b"x", engine, dpi=0)
    with pytest.raises(ValueError):
        ocr_pdf(b"x", engine, max_pixels=0)


def test_pdf_error_is_base(engine, text_pdf):
    with pytest.raises(PdfError):
        ocr_pdf(b"nope", engine)
