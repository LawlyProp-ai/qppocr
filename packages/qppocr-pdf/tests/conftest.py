"""Shared fixtures: synthetic PDFs built at test time (no committed binaries).

Every fixture is materialized in memory / a pytest tmp dir, so the repository
carries no binary test assets.
"""

from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SCAN_TEXT = "SCANNED TEXT 123"
TEXT_PAGE_TEXT = "Hello PDF World"


def _assemble(objects: dict[int, bytes]) -> bytes:
    """Assemble a minimal PDF with a correct xref table.

    ``objects`` maps object number -> the bytes between ``N 0 obj`` and
    ``endobj`` (including any ``stream``/``endstream`` body).
    """
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in sorted(objects):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\n"
        f"startxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(out)


def _stream(data: bytes) -> bytes:
    return b"<</Length %d>>\nstream\n" % len(data) + data + b"\nendstream"


def build_text_pdf() -> bytes:
    """One page with a real Helvetica text layer and no images."""
    content = b"BT /F1 24 Tf 72 700 Td (%s) Tj ET" % TEXT_PAGE_TEXT.encode()
    return _assemble(
        {
            1: b"<</Type/Catalog/Pages 2 0 R>>",
            2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            3: (
                b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
                b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>"
            ),
            4: b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
            5: _stream(content),
        }
    )


def _small_jpeg() -> bytes:
    image = Image.new("RGB", (64, 64), (30, 144, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def build_image_pdf(*, full_page: bool) -> bytes:
    """One page with a text layer plus an image XObject.

    ``full_page=True`` covers the page (a scan-like mixed page); ``False`` places
    a small logo that must not force OCR.
    """
    width_pt, height_pt = (612.0, 792.0) if full_page else (72.0, 72.0)
    content = b"q %.2f 0 0 %.2f 0 0 cm /Im0 Do Q\n" % (width_pt, height_pt)
    content += b"BT /F1 24 Tf 72 700 Td (%s) Tj ET" % TEXT_PAGE_TEXT.encode()
    jpeg = _small_jpeg()
    image_obj = (
        b"<</Type/XObject/Subtype/Image/Width 64/Height 64/ColorSpace/DeviceRGB"
        b"/BitsPerComponent 8/Filter/DCTDecode/Length %d>>\nstream\n" % len(jpeg)
        + jpeg
        + b"\nendstream"
    )
    return _assemble(
        {
            1: b"<</Type/Catalog/Pages 2 0 R>>",
            2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            3: (
                b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
                b"/Resources<</Font<</F1 4 0 R>>/XObject<</Im0 6 0 R>>>>"
                b"/Contents 5 0 R>>"
            ),
            4: b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
            5: _stream(content),
            6: image_obj,
        }
    )


def build_scan_pdf() -> bytes:
    """One image-only page (legible drawn text) with no text layer."""
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    font = (
        ImageFont.truetype(str(FONT_PATH), 72)
        if FONT_PATH.exists()
        else ImageFont.load_default()
    )
    draw.text((120, 300), SCAN_TEXT, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PDF", resolution=200)
    return buffer.getvalue()


def build_oversized_pdf() -> bytes:
    """One empty page far larger than any sane raster cap."""
    return _assemble(
        {
            1: b"<</Type/Catalog/Pages 2 0 R>>",
            2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            3: b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 20000 20000]>>",
        }
    )


def build_deeply_nested_pdf(depth: int = 12) -> bytes:
    """One page whose image sits inside ``depth`` nested Form XObjects.

    ``depth`` exceeds ``backend.MAX_OBJECT_DEPTH``, so a correctly depth-bounded
    traversal cannot reach the image and must still terminate cleanly.
    """
    jpeg = _small_jpeg()
    image_number = 5 + depth
    image_obj = (
        b"<</Type/XObject/Subtype/Image/Width 64/Height 64/ColorSpace/DeviceRGB"
        b"/BitsPerComponent 8/Filter/DCTDecode/Length %d>>\nstream\n" % len(jpeg)
        + jpeg
        + b"\nendstream"
    )
    objects: dict[int, bytes] = {
        1: b"<</Type/Catalog/Pages 2 0 R>>",
        2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        3: (
            b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            b"/Resources<</XObject<</Fm0 4 0 R>>>>/Contents 5 0 R>>"
        ),
        5: _stream(b"q 612 0 0 792 0 0 cm /Fm0 Do Q"),
        image_number: image_obj,
    }
    for i in range(depth):
        if i == depth - 1:
            name, target = b"Im0", image_number
        else:
            name, target = b"Fm%d" % (i + 1), 4 + i + 1
        content = b"q 612 0 0 792 0 0 cm /%s Do Q" % name
        objects[4 + i] = (
            b"<</Type/XObject/Subtype/Form/BBox[0 0 612 792]"
            b"/Resources<</XObject<</%s %d 0 R>>>>/Length %d>>\nstream\n"
            % (name, target, len(content))
            + content
            + b"\nendstream"
        )
    return _assemble(objects)


# --- Minimal RC4 (R2/V1) standard-security-handler PDF ----------------------

_PAD = bytes(
    [
        0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56,
        0xFF, 0xFA, 0x01, 0x08, 0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80,
        0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
    ]
)


def _pad_password(password: bytes) -> bytes:
    password = password[:32]
    return password + _PAD[: 32 - len(password)]


def _rc4(key: bytes, data: bytes) -> bytes:
    state = list(range(256))
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) & 0xFF
        state[i], state[j] = state[j], state[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) & 0xFF
        j = (j + state[i]) & 0xFF
        state[i], state[j] = state[j], state[i]
        out.append(byte ^ state[(state[i] + state[j]) & 0xFF])
    return bytes(out)


def build_encrypted_pdf(user: bytes = b"secret", owner: bytes = b"owner") -> bytes:
    """A V1/R2 PDF requiring ``user`` to open (no password -> load error)."""
    permissions = -4
    doc_id = b"\x01" * 16

    owner_key = hashlib.md5(_pad_password(owner)).digest()[:5]
    owner_entry = _rc4(owner_key, _pad_password(user))
    user_key = hashlib.md5(
        _pad_password(user) + owner_entry + struct.pack("<i", permissions) + doc_id
    ).digest()[:5]
    user_entry = _rc4(user_key, _PAD)

    encrypt = (
        b"<</Filter/Standard/V 1/R 2/Length 40/P -4/O <"
        + owner_entry.hex().encode()
        + b">/U <"
        + user_entry.hex().encode()
        + b">>>"
    )
    objects: dict[int, bytes] = {
        1: b"<</Type/Catalog/Pages 2 0 R>>",
        2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        3: b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>",
        4: encrypt,
    }
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in sorted(objects):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    plain_id = doc_id.hex().encode()
    out += (
        b"trailer\n<</Size 5/Root 1 0 R/Encrypt 4 0 R/ID [<"
        + plain_id
        + b"><"
        + plain_id
        + b">]>>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF"
    )
    return bytes(out)


@pytest.fixture(scope="session")
def text_pdf() -> bytes:
    return build_text_pdf()


@pytest.fixture(scope="session")
def scan_pdf() -> bytes:
    return build_scan_pdf()


@pytest.fixture(scope="session")
def mixed_pdf() -> bytes:
    return build_image_pdf(full_page=True)


@pytest.fixture(scope="session")
def logo_pdf() -> bytes:
    return build_image_pdf(full_page=False)


@pytest.fixture(scope="session")
def oversized_pdf() -> bytes:
    return build_oversized_pdf()


@pytest.fixture(scope="session")
def deep_nested_pdf() -> bytes:
    return build_deeply_nested_pdf()


@pytest.fixture(scope="session")
def encrypted_pdf() -> bytes:
    return build_encrypted_pdf()
