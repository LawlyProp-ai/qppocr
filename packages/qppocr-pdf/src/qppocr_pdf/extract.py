"""Public PDF -> per-page text/OCR extraction.

Each page is routed either to its embedded text layer or to OCR of a rasterized
page. OCR geometry is scaled from pixels back to PDF points via ``72 / dpi``.
All failures (malformed, encrypted, oversized, backend) raise a :class:`PdfError`
subclass; no partial result list is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pypdfium2 as pdfium

from . import backend
from .classify import decide

Point = tuple[float, float]
Pts = tuple[Point, ...]

#: Default rasterization resolution, in DPI.
DEFAULT_DPI = 200
#: Default image-coverage fraction above which a text page is OCR'd anyway.
DEFAULT_COVERAGE_THRESHOLD = 0.5
#: Default cap on rendered pixels per page (width * height).
DEFAULT_MAX_PIXELS = 40_000_000


class PdfError(Exception):
    """Base class for PDF ingestion failures."""


class MalformedPdfError(PdfError):
    """The input could not be parsed as a PDF."""


class EncryptedPdfError(PdfError):
    """The document is encrypted and cannot be read."""


class PageTooLargeError(PdfError):
    """A page would rasterize to more pixels than the configured cap."""


@dataclass(frozen=True)
class Char:
    """One recognized character and its quad, in PDF points."""

    text: str
    pts: Pts


@dataclass(frozen=True)
class Line:
    """One recognized text line, in PDF points."""

    text: str
    confidence: float
    pts: Pts
    chars: tuple[Char, ...] = ()


@dataclass
class PageResult:
    """One page's payload: ``kind`` is ``"text"`` or ``"ocr"``."""

    kind: str
    text: str
    lines: list[Line]
    width_pt: float
    height_pt: float
    scale: float
    dpi: int


def _is_password_error(exc: BaseException) -> bool:
    code = getattr(exc, "err_code", None)
    if code is not None and code == pdfium.raw.FPDF_ERR_PASSWORD:
        return True
    return "password" in str(exc).lower()


def _open_document(source: Any) -> Any:
    """Open ``source`` (path or bytes), mapping failures to clear errors."""
    try:
        doc = pdfium.PdfDocument(source)
    except pdfium.PdfiumError as exc:
        if _is_password_error(exc):
            raise EncryptedPdfError("PDF is encrypted and requires a password") from exc
        raise MalformedPdfError(f"cannot parse PDF input: {exc}") from exc
    except (TypeError, FileNotFoundError):
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MalformedPdfError(f"cannot parse PDF input: {exc}") from exc

    # PDFium only reports a security handler when the document is encrypted.
    try:
        revision = pdfium.raw.FPDF_GetSecurityHandlerRevision(doc)
    except Exception:  # noqa: BLE001 - defensive: symbol may be unavailable; treat as unencrypted  # pragma: no cover
        revision = -1
    if revision >= 0:
        doc.close()
        raise EncryptedPdfError(f"PDF is encrypted (security handler revision {revision})")

    return doc


def _scaled_line(line: Any, scale: float) -> Line:
    pts: Pts = tuple((float(x) * scale, float(y) * scale) for x, y in line.pts)
    chars: list[Char] = []
    for char in getattr(line, "chars", None) or ():
        char_pts: Pts = tuple((float(x) * scale, float(y) * scale) for x, y in char.pts)
        chars.append(Char(text=char.text, pts=char_pts))
    return Line(
        text=line.text,
        confidence=float(line.confidence),
        pts=pts,
        chars=tuple(chars),
    )


def _process_page(
    doc: Any,
    page: Any,
    index: int,
    engine: Any,
    *,
    dpi: int,
    scale: float,
    coverage_threshold: float,
    force: str,
    max_pixels: int,
) -> PageResult:
    width_pt, height_pt = (float(v) for v in page.get_size())

    if force == "text":
        kind = "text"
    elif force == "ocr":
        kind = "ocr"
    else:
        has_text, coverage = backend.page_signals(doc, page)
        kind = decide(has_text, coverage, threshold=coverage_threshold, force=force)

    if kind == "text":
        return PageResult(
            kind="text",
            text=backend.extract_text(page),
            lines=[],
            width_pt=width_pt,
            height_pt=height_pt,
            scale=1.0,
            dpi=dpi,
        )

    width_px = math.ceil(width_pt * dpi / 72.0)
    height_px = math.ceil(height_pt * dpi / 72.0)
    if width_px * height_px > max_pixels:
        raise PageTooLargeError(
            f"page {index + 1} would rasterize to {width_px}x{height_px} px "
            f"({width_px * height_px} > max_pixels={max_pixels})"
        )

    png = backend.render_page_png(page, dpi)
    result = engine.run(png)
    lines = [_scaled_line(line, scale) for line in result.lines]
    return PageResult(
        kind="ocr",
        text="\n".join(line.text for line in lines),
        lines=lines,
        width_pt=width_pt,
        height_pt=height_pt,
        scale=scale,
        dpi=dpi,
    )


def ocr_pdf(
    source: Any,
    engine: Any,
    *,
    dpi: int = DEFAULT_DPI,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    force: str = "auto",
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> list[PageResult]:
    """Extract every page of ``source`` (path or bytes) as a :class:`PageResult`.

    ``engine`` is a ``qppocr.Engine`` (anything exposing ``run(png_bytes)``).
    ``force`` is ``"auto" | "text" | "ocr"`` and bypasses classification.
    Pages whose rasterized pixel count exceeds ``max_pixels`` raise
    :class:`PageTooLargeError`.
    """
    if force not in ("auto", "text", "ocr"):
        raise ValueError(f"force must be 'auto', 'text' or 'ocr', got {force!r}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi!r}")
    if max_pixels <= 0:
        raise ValueError(f"max_pixels must be positive, got {max_pixels!r}")

    scale = 72.0 / dpi
    doc = _open_document(source)
    try:
        page_count = len(doc)
        results: list[PageResult] = []
        for index in range(page_count):
            page = doc.get_page(index)
            try:
                results.append(
                    _process_page(
                        doc,
                        page,
                        index,
                        engine,
                        dpi=dpi,
                        scale=scale,
                        coverage_threshold=coverage_threshold,
                        force=force,
                        max_pixels=max_pixels,
                    )
                )
            except PdfError:
                raise
            except Exception as exc:
                raise PdfError(f"failed to process page {index + 1}: {exc}") from exc
        return results
    finally:
        doc.close()
