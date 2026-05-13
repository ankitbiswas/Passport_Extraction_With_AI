"""Load input files (PDF or image), convert pages to base64-encoded PNG strings."""

import base64
import io
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def find_input_files(directory: str | Path) -> list[Path]:
    """Return all supported input files found in *directory* (non-recursive)."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(sorted(dir_path.glob(f"*{ext}")))
        files.extend(sorted(dir_path.glob(f"*{ext.upper()}")))
    return sorted(set(files))


def file_to_base64_images(
    file_path: str | Path, dpi: int = 200, max_dim: int | None = 1200
) -> list[str]:
    """Convert a PDF or image file to a list of base64-encoded PNG strings.

    For PDFs: each page becomes one image.
    For images: the single image becomes a one-element list.

    If *max_dim* is set, images larger than that on their longest side are
    downscaled before encoding (keeps total payload under API limits).
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return _pdf_to_base64_images(file_path, dpi=dpi, max_dim=max_dim)
    elif ext in {".jpg", ".jpeg", ".png"}:
        return _image_to_base64(file_path, max_dim=max_dim)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _resize_if_needed(img: Image.Image, max_dim: int | None) -> Image.Image:
    if max_dim is None:
        return img
    w, h = img.size
    longest = max(w, h)
    if longest <= max_dim:
        return img
    ratio = max_dim / longest
    new_size = (int(w * ratio), int(h * ratio))
    return img.resize(new_size, Image.LANCZOS)


def _pdf_to_base64_images(
    pdf_path: Path, dpi: int = 200, max_dim: int | None = 1200
) -> list[str]:
    images: list[Image.Image] = convert_from_path(pdf_path, dpi=dpi)
    encoded: list[str] = []
    for img in images:
        img = _resize_if_needed(img, max_dim)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return encoded


def _image_to_base64(
    image_path: Path, max_dim: int | None = 1200
) -> list[str]:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = _resize_if_needed(img, max_dim)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return [base64.b64encode(buf.getvalue()).decode("utf-8")]
