"""pypdfium2 backend: per-page text/image signals, rasterization and text extraction.

PDF input is untrusted. Object traversal is depth-bounded and only the image
object type is requested, so a pathological document cannot walk an unbounded
object graph.
"""

from __future__ import annotations

import io
from typing import Any

import pypdfium2 as pdfium

#: Image page object constant (``FPDF_PAGEOBJ_IMAGE == 3``).
_IMAGE = pdfium.raw.FPDF_PAGEOBJ_IMAGE

#: Upper bound on Form XObject / object nesting when collecting image objects.
MAX_OBJECT_DEPTH = 8


def page_signals(doc: Any, page: Any) -> tuple[bool, float]:
    """Return ``(has_text, coverage)`` for one page.

    ``has_text`` is true when the page has any character in its text layer.
    ``coverage`` is the fraction of the page area covered by image objects
    (summed intersection with the page box, clamped to ``[0, 1]``).
    """
    has_text = page.get_textpage().count_chars() > 0

    left, bottom, right, top = (float(v) for v in page.get_bbox())
    page_area = abs((right - left) * (top - bottom))
    if page_area <= 0.0:
        return has_text, 0.0

    covered = 0.0
    for obj in page.get_objects(filter=[_IMAGE], max_depth=MAX_OBJECT_DEPTH):
        ol, ob, oright, otop = (float(v) for v in obj.get_bounds())
        width = max(0.0, min(oright, right) - max(ol, left))
        height = max(0.0, min(otop, top) - max(ob, bottom))
        covered += width * height

    return has_text, min(1.0, max(0.0, covered / page_area))


def render_page_png(page: Any, dpi: int) -> bytes:
    """Rasterize ``page`` at ``dpi`` and return RGB PNG bytes."""
    scale = dpi / 72.0
    image = page.render(scale=scale).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_text(page: Any) -> str:
    """Return the page's embedded text with CRLF/CR normalized to LF."""
    text: str = page.get_textpage().get_text_bounded()
    return text.replace("\r\n", "\n").replace("\r", "\n")
