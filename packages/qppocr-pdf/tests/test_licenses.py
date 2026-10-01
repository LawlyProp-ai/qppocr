"""License-closure check: pure predicate plus an installed-closure smoke test."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_licenses.py"
_spec = importlib.util.spec_from_file_location("check_licenses", _SCRIPT)
assert _spec and _spec.loader
check_licenses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_licenses)


def test_permissive_licenses_pass():
    entries = [
        ("a", "MIT"),
        ("b", "BSD-3-Clause"),
        ("c", "BSD-2-Clause"),
        ("d", "Apache-2.0"),
        ("e", "ISC"),
        ("f", "HPND"),
        ("g", "PSF-2.0"),
        ("h", "0BSD"),
        ("i", "MIT-0"),
        ("j", "MIT-CMU"),
        ("k", "Apache-2.0 OR BSD-3-Clause"),
    ]
    assert check_licenses.check_licenses(entries) == []


@pytest.mark.parametrize(
    "license_text",
    ["AGPL-3.0", "GPL-2.0", "LGPL-2.1", "SSPL-1.0", "MPL-2.0"],
)
def test_copyleft_licenses_are_violations(license_text):
    problems = check_licenses.check_licenses([("bad", license_text)])
    assert len(problems) == 1
    assert "copyleft" in problems[0]


@pytest.mark.parametrize("license_text", ["", "UNKNOWN", "Proprietary", "Look at LICENSE"])
def test_unknown_license_needs_review(license_text):
    problems = check_licenses.check_licenses([("mystery", license_text)])
    assert len(problems) == 1
    assert "needs review" in problems[0]


def test_requirement_parsing_ignores_extras_and_markers():
    assert check_licenses._requirement_name('sphinx>=8.2; extra == "docs"') == "sphinx"
    assert check_licenses._is_extra('sphinx>=8.2; extra == "docs"') is True
    assert check_licenses._is_extra("pypdfium2==5.13.0") is False


def test_installed_closure_is_permissive():
    import importlib.metadata

    try:
        importlib.metadata.metadata("qppocr-pdf")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("qppocr-pdf is not installed")

    entries = check_licenses.installed_closure("qppocr-pdf")
    names = {name for name, _ in entries}
    assert {"qppocr-pdf", "qppocr", "pypdfium2", "pillow"} <= names
    assert check_licenses.check_licenses(entries) == []
