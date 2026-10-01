"""Unit tests for the pure routing predicate."""

import pytest

from qppocr_pdf import decide


def test_text_only_page_is_text():
    assert decide(has_text=True, coverage=0.0) == "text"


def test_scan_page_is_ocr():
    assert decide(has_text=False, coverage=1.0) == "ocr"


def test_mixed_page_above_threshold_is_ocr():
    assert decide(has_text=True, coverage=0.9, threshold=0.5) == "ocr"


def test_logo_below_threshold_stays_text():
    assert decide(has_text=True, coverage=0.01, threshold=0.5) == "text"


def test_force_text_bypasses_classification():
    assert decide(has_text=False, coverage=1.0, force="text") == "text"


def test_force_ocr_bypasses_classification():
    assert decide(has_text=True, coverage=0.0, force="ocr") == "ocr"


def test_custom_threshold():
    assert decide(has_text=True, coverage=0.6, threshold=0.75) == "text"
    assert decide(has_text=True, coverage=0.8, threshold=0.75) == "ocr"


def test_invalid_force_raises():
    with pytest.raises(ValueError):
        decide(has_text=True, coverage=0.0, force="sometimes")
