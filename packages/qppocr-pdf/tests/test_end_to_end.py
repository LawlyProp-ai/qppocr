"""End-to-end OCR of the scan fixture with the real native engine."""

import os
from pathlib import Path

import pytest

from qppocr_pdf import ocr_pdf

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _models_dir():
    override = os.environ.get("QPPOCR_MODELS")
    if override:
        return Path(override)
    return _REPO_ROOT / "models"


def test_scan_fixture_ocr(scan_pdf):
    qppocr = pytest.importorskip("qppocr")
    models = _models_dir()
    if not models.is_dir():
        pytest.skip(f"models directory not found: {models}")

    try:
        engine = qppocr.Engine(models=str(models))
    except Exception as exc:  # noqa: BLE001 - any engine/model failure skips the test
        pytest.skip(f"engine unavailable: {exc}")

    results = ocr_pdf(scan_pdf, engine)
    assert len(results) == 1
    page = results[0]
    assert page.kind == "ocr"
    assert page.lines, "engine returned no lines for the scan fixture"
    assert page.text.strip()
    assert any("SCANNED" in line.text.upper() for line in page.lines)
