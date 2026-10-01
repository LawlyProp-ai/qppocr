#!/usr/bin/env python3
"""Permissive-license closure check for the ``qppocr-pdf`` distribution.

``check_licenses`` is a pure function over ``(name, license)`` pairs so it can be
unit-tested. The CLI walks the installed runtime dependency closure
(``Requires-Dist`` transitively from ``qppocr-pdf``) and reports every copyleft
or unrecognized license. Unknown/empty metadata is reported for review, never
silently passed.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from typing import Any

# Compacted (non-alphanumerics removed, upper-cased) substrings that make a
# license permissive. See Project Article: permissive-only dependencies.
PERMISSIVE = (
    "MIT",  # covers MIT, MIT-0, MIT-CMU
    "BSD2CLAUSE",
    "BSD3CLAUSE",
    "APACHE20",
    "ISC",
    "HPND",
    "PSF",
    "0BSD",
)

# Compacted substrings that mark a copyleft license as a violation.
COPYLEFT = ("AGPL", "LGPL", "GPL", "SSPL", "MPL")

ROOT = "qppocr-pdf"


def _compact(license_text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", license_text or "").upper()


def check_licenses(entries: list[tuple[str, str]]) -> list[str]:
    """Return one problem string per non-permissive entry (empty list = clean).

    Copyleft entries are violations; empty/unknown licenses are returned as
    ``"needs review"`` problems rather than passing silently.
    """
    problems: list[str] = []
    for name, license_text in entries:
        compact = _compact(license_text)
        if not compact or "UNKNOWN" in compact:
            problems.append(f"needs review: {name}: license metadata is empty/unknown")
        elif any(token in compact for token in COPYLEFT):
            problems.append(f"{name}: copyleft license {license_text!r} is not allowed")
        elif not any(token in compact for token in PERMISSIVE):
            problems.append(f"needs review: {name}: license {license_text!r} is not in the allowlist")
    return problems


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return _canonical(match.group(1)) if match else ""


def _is_extra(requirement: str) -> bool:
    _, sep, marker = requirement.partition(";")
    return bool(sep) and "extra" in marker


def _license_of(meta: Any) -> str:
    license_text = meta.get("License-Expression") or meta.get("License") or ""
    compact = _compact(license_text)
    if not compact or "UNKNOWN" in compact:
        for classifier in meta.get_all("Classifier") or []:
            if classifier.startswith("License ::"):
                license_text = classifier.split("::")[-1].strip()
                break
    return license_text.strip()


def installed_closure(root: str = ROOT) -> list[tuple[str, str]]:
    """Return ``(name, license)`` for ``root`` and its transitive runtime deps."""
    seen: dict[str, str] = {}
    stack = [_canonical(root)]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            meta = metadata.metadata(name)
        except metadata.PackageNotFoundError:
            seen[name] = ""
            continue
        seen[name] = _license_of(meta)
        for requirement in metadata.requires(name) or []:
            if _is_extra(requirement):
                continue
            dependency = _requirement_name(requirement)
            if dependency:
                stack.append(dependency)
    return sorted(seen.items())


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else ROOT
    entries = installed_closure(root)
    problems = check_licenses(entries)
    print(f"checked {len(entries)} distributions from {root!r}:")
    for name, license_text in entries:
        print(f"  {name}: {license_text or '<unknown>'}")
    if problems:
        print("\nlicense check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nlicense check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
