"""Main entry point — walk a directory of PDFs, extract structured data, print results."""

import datetime
import json
from pathlib import Path

from pdf_processor import find_input_files, file_to_base64_images
from ocr import ocr_pages
from llm_extractor import extract
from schema import Passport

INPUT_DIR = Path(__file__).parent / "input_pdfs"
OUTPUT_DIR = Path(__file__).parent / "output"
DPI = 200


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = OUTPUT_DIR / ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    files = find_input_files(INPUT_DIR)

    if not files:
        print(f"No supported files found in {INPUT_DIR}")
        return

    for file_path in files:
        print(f"Processing: {file_path.name}")
        pages_b64 = file_to_base64_images(file_path, dpi=DPI)
        ocr_texts = ocr_pages(pages_b64, source_stem=file_path.stem)
        extracted = extract(ocr_texts, pages_b64, Passport)
        print(f"  -> {json.dumps(extracted.model_dump(), indent=2)}")

        output_file = run_output_dir / f"{file_path.stem}.json"
        output_file.write_text(
            json.dumps(extracted.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Output: {output_file}")

    print(f"\nProcessed {len(files)} file(s). Output: {run_output_dir}")


if __name__ == "__main__":
    main()
