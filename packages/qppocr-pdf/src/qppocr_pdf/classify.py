"""Pure per-page routing decision: embedded text layer vs. OCR."""

from __future__ import annotations

FORCES = ("auto", "text", "ocr")


def decide(has_text: bool, coverage: float, threshold: float = 0.5, force: str = "auto") -> str:
    """Return ``"text"`` or ``"ocr"`` for one page.

    ``force`` bypasses classification: ``"text"`` always returns the embedded
    layer, ``"ocr"`` always rasterizes. Otherwise a page is treated as a scan
    when it has no text layer, or when an image covers *more than* ``threshold``
    of the page (a logo therefore does not force OCR).
    """
    if force not in FORCES:
        raise ValueError(f"force must be one of {FORCES}, got {force!r}")
    if force == "text":
        return "text"
    if force == "ocr":
        return "ocr"
    if not has_text:
        return "ocr"
    if coverage > threshold:
        return "ocr"
    return "text"
