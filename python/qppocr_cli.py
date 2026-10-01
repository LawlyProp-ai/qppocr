"""Run the qppocr Rust OCR engine from Python.

There are no Python bindings in this repo (pure Rust, zero FFI), so this drives
the built `qppocr` CLI as a subprocess and parses its `--json` output.

Setup:
    cargo build --release -p qppocr-cli          # see notes in fetch_models()
    python3 python/qppocr.py fetch tiny small    # download weights + dicts

Usage:
    from qppocr_cli import ocr
    for line in ocr("receipt.png")["lines"]:
        print(line["confidence"], line["text"], line["box"])
"""

import json
import shutil
import subprocess
import sys
import urllib.request

HF = "https://huggingface.co/PaddlePaddle/{repo}/resolve/main/inference.onnx"
# (repo, sha256) per crates/qppocr/src/models.rs; cls/dict are not SHA-checked by the engine.
MODELS = {
    "tiny": {
        "det": ("PP-OCRv6_tiny_det_onnx", "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8"),
        "rec": ("PP-OCRv6_tiny_rec_onnx", "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6"),
    },
    "small": {
        "det": ("PP-OCRv6_small_det_onnx", "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e"),
        "rec": ("PP-OCRv6_small_rec_onnx", "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634"),
    },
    "medium": {
        "det": ("PP-OCRv6_medium_det_onnx", "eb13b44b25bb36f89528b68720af8a61d9cf381176107f465db1757b65d086e1"),
        "rec": ("PP-OCRv6_medium_rec_onnx", "9c09abf0957f7968c7586464b7397b84ad2387a0497a351af40e9acc71b673ba"),
    },
}
CLS = ("PP-LCNet_x0_25_textline_ori_onnx", "dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2")


def ocr(images, models="models", tier="small", preset=None, binary=None, extra=()):
    """OCR one image (str) or a list of images; returns the parsed JSON.

    Single image -> dict; list -> list of dicts (same order). One CLI call per
    invocation, so a batch loads the weights once (pass a list, don't loop).
    """
    single = isinstance(images, (str, bytes))
    paths = [images] if single else list(images)
    cmd = [binary or shutil.which("qppocr") or "qppocr", *map(str, paths),
           "--models", str(models), "--tier", tier, "--json"]
    if preset:
        cmd += ["--preset", preset]
    cmd += extra
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    return data[0] if single else data


def _chars_from_yml(text):
    """Pull PostProcess.character_dict out of a Paddle inference.yml (no PyYAML)."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("character_dict:"))
    chars = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s:
            continue
        if not s.startswith("-"):
            break
        v = s[1:].strip()
        if len(v) >= 2 and v[0] == v[-1] == "'":
            v = v[1:-1].replace("''", "'")
        elif len(v) >= 2 and v[0] == v[-1] == '"':
            v = json.loads(v)
        chars.append(v)
    return chars


def fetch_models(tiers=("tiny", "small"), dest="models"):
    """Download weights + extract dict.txt from inference.yml into `dest`."""
    import hashlib
    import os

    os.makedirs(dest, exist_ok=True)

    def get(url):
        with urllib.request.urlopen(url, timeout=300) as r:
            return r.read()

    for tier in tiers:
        d = os.path.join(dest, tier)
        os.makedirs(d, exist_ok=True)
        for kind, (repo, sha) in MODELS[tier].items():
            path = os.path.join(d, f"{kind}.onnx")
            if os.path.exists(path):
                print(f"have {path}")
            else:
                print(f"download {path} ...")
                data = get(HF.format(repo=repo))
                assert hashlib.sha256(data).hexdigest() == sha, f"{path}: SHA-256 mismatch"
                open(path, "wb").write(data)
        yml = get(f"https://huggingface.co/PaddlePaddle/{MODELS[tier]['rec'][0]}/raw/main/inference.yml").decode()
        chars = _chars_from_yml(yml)
        print(f"{tier}: dict {len(chars)} chars")
        open(os.path.join(d, "dict.txt"), "w", encoding="utf-8").write("\n".join(chars) + "\n")
        if tier in ("small", "medium"):
            open(os.path.join(dest, "dict.txt"), "w", encoding="utf-8").write("\n".join(chars) + "\n")

    cls_path = os.path.join(dest, "cls.onnx")
    if not os.path.exists(cls_path):
        print(f"download {cls_path} ...")
        open(cls_path, "wb").write(get(HF.format(repo=CLS[0])))
    print("done")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        fetch_models(sys.argv[2:] or ("tiny", "small"))
    else:
        print(__doc__)
