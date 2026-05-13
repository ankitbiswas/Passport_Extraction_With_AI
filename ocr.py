"""OCR with pytesseract + disk cache so we never OCR the same page twice."""

import base64
import hashlib
import io
import json
from pathlib import Path

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CACHE_DIR = Path(__file__).parent / "cache"


def _cache_path(page_b64: str, source_stem: str = "") -> Path:
    digest = hashlib.sha256(page_b64.encode()).hexdigest()
    prefix = f"{source_stem}_" if source_stem else ""
    return CACHE_DIR / f"{prefix}{digest}.json"


def _read_cache(page_b64: str, source_stem: str = "") -> str | None:
    path = _cache_path(page_b64, source_stem)
    if path.exists():
        # print("cache exists for page, loading from disk")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["text"]
    return None


def _write_cache(page_b64: str, text: str, source_stem: str = "") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(page_b64, source_stem).write_text(
        json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
    )


def ocr_page(page_b64: str, source_stem: str = "") -> str:
    """Run pytesseract on a base64-encoded PNG. Results are cached to disk."""
    cached = _read_cache(page_b64, source_stem)
    if cached is not None:
        return cached

    image_bytes = base64.b64decode(page_b64)
    img = Image.open(io.BytesIO(image_bytes))

    text = pytesseract.image_to_string(img)
    _write_cache(page_b64, text, source_stem)
    return text


def ocr_pages(pages_b64: list[str], source_stem: str = "") -> list[str]:
    """OCR a list of base64-encoded page images."""
    return [ocr_page(p, source_stem) for p in pages_b64]
