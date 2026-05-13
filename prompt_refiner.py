"""Load prompts, images, OCR cache, and comparison results — ready for LLM feedback loop."""

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from pdf_processor import find_input_files, file_to_base64_images
from ocr import ocr_pages
from llm_extractor import _load_user_prompt

load_dotenv()

INPUT_DIR = Path(__file__).parent / "input_pdfs"
COMPARISON_DIR = Path(__file__).parent / "comparison_results"
ERROR_REPORT_DIR = Path(__file__).parent / "error_report"
SYSTEM_PROMPT = "You are a precise document data extraction assistant. Extract only the requested fields. Return valid JSON matching the schema exactly. If a field is not found, use an empty string."

# Set this to a specific timestamp folder name to override auto-latest.
# Example: "2026-05-11_14-30-22"
COMPARISON_TIMESTAMP: str | None = None

FIELD_ORDER = [
    "first_name", "last_name", "passport_number", "nationality",
    "date_of_birth", "date_of_expiry", "issuing_country", "issuing_authority",
]

REFINER_SYSTEM_PROMPT = (
    "You are an expert prompt engineer specializing in document extraction systems. "
    "You analyze extraction errors and rewrite prompts to improve accuracy."
)


def _latest_dir(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    dirs = sorted(
        [d for d in parent.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


def load_comparison_results() -> tuple[pd.DataFrame, Path]:
    """Load comparison_results.xlsx from the chosen or latest timestamp folder.

    Returns (DataFrame, folder_path).
    """
    if COMPARISON_TIMESTAMP:
        folder = COMPARISON_DIR / COMPARISON_TIMESTAMP
    else:
        folder = _latest_dir(COMPARISON_DIR)

    if folder is None or not folder.exists():
        raise FileNotFoundError(
            f"Comparison folder not found. Set COMPARISON_TIMESTAMP or run analysis.py first."
        )

    path = folder / "comparison_results.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"comparison_results.xlsx not found in {folder}")

    print(f"Loading comparison results from: {folder.name}")
    return pd.read_excel(path, dtype={"passport_number_gt": str, "passport_number_extracted": str}), folder


def build_error_report(df: pd.DataFrame) -> list[dict]:
    """Build per-field error report as list of dicts (JSON-friendly)."""
    report = []

    for field in FIELD_ORDER:
        compare_col = f"{field}_compare"
        gt_col = f"{field}_gt"
        ext_col = f"{field}_extracted"

        # Error rows: FP, FN, MISTAKE
        error_mask = df[compare_col].isin(["FP", "FN", "MISTAKE"])
        error_rows = df.loc[error_mask, ["source_file", gt_col, ext_col, compare_col]]

        errors = []
        for _, row in error_rows.iterrows():
            errors.append({
                "source_file": row["source_file"],
                "gt": str(row[gt_col]) if pd.notna(row[gt_col]) else "",
                "extracted": str(row[ext_col]) if pd.notna(row[ext_col]) else "",
                "type": row[compare_col],
            })

        # Correct samples: up to 3 TP rows
        tp_mask = df[compare_col] == "TP"
        tp_rows = df.loc[tp_mask, ["source_file", gt_col, ext_col]].head(3)
        correct_samples = []
        for _, row in tp_rows.iterrows():
            correct_samples.append({
                "source_file": row["source_file"],
                "gt": str(row[gt_col]) if pd.notna(row[gt_col]) else "",
                "extracted": str(row[ext_col]) if pd.notna(row[ext_col]) else "",
            })

        report.append({
            "field": field,
            "error_count": len(errors),
            "correct_count": int(tp_mask.sum()),
            "errors": errors,
            "correct_samples": correct_samples,
        })

    return report


def build_refiner_prompt(current_user_prompt: str, error_report: list[dict]) -> str:
    """Build the meta-prompt that asks the LLM to analyze errors and refine the prompt."""
    error_json = json.dumps(error_report, indent=2, ensure_ascii=False)

    return f"""You are an expert prompt engineer for a passport data extraction system. Your task is to analyze extraction errors and rewrite the user prompt to eliminate those errors.

=== CURRENT USER PROMPT ===
{current_user_prompt}

=== ERROR REPORT (JSON) ===
{error_json}

=== INSTRUCTIONS ===
1. Study the error report carefully. For each field, you have:
   - "errors": rows where extraction failed (FP, FN, MISTAKE) — GT vs extracted value with error type. So in these cases look what the model produced versus what was the GT (the ground truth expectations).
   - "correct_samples": rows where extraction succeeded (TP) — GT vs extracted value match exactly.
   - The document images and OCR text for these files are attached below — examine them to see what the extraction model saw.

2. Compare the error cases against the correct cases. For each field, identify the PATTERN: what is DIFFERENT between the errors and the correct samples? What systematic root cause explains the failures? Look at the GT vs extracted values, the error types, and the visual/OCR evidence.
   Also, even in the error cases, you can look at the GT (Ground Truth) values to have an understanding of how the fields are expected to be formatted & extracted so that you are able to find a clue as to how the prompt could be updated to get the model to produce the expected GT values instead of the current errors.

   HINT: Pay close attention to character-level differences between GT and extracted values. For example, if GT has plain ASCII but extracted has accented/diacritic characters (or vice versa), this reveals a normalization gap. Similarly, if GT uses a specific encoding convention (e.g., "OE" for "Ø", "AE" for "Æ", "A" for "Ä"), the prompt must instruct the model to follow that convention.

3. CRITICAL CONSTRAINTS:
   - Keep the same overall structure and field list (first_name, last_name, passport_number, nationality, date_of_birth, date_of_expiry, issuing_country, issuing_authority).
   - YOU CAN ADD PASSPORT RELATED KNOWLEDGE OF DIFFERENT COUNTRIES IF YOU INFER THAT THE MODEL LACKS SOME KNOWLEDGE ABOUT CERTAIN FORMATS OR CONVENTIONS (e.g., how certain fields are typically formatted in passports from different countries).
   - Try to have the model infer the underlying rules and patterns rather than just memorizing specific examples. The refined prompt should guide the model to apply the correct logic to any passport, not just the ones in the current error set.
   - Try to add more general Guardrails like Diacritics Handling you can use your knowledge to pass broader information of diacritics and their corresponding ASCII characters, etc. Example: "If the GT has 'O' but the extracted has 'Ø', this indicates that the model is producing accented characters. To fix this, add a guardrail to the prompt: 'When extracting names, if you encounter accented characters, convert them to their plain ASCII equivalents (e.g., Ø -> O, Æ -> AE, Ä -> A) to match the GT format.'"
   - ADD MORE 'CRITICAL' points if you identify any other common root causes in the errors that can be fixed via prompt instructions.
   - Fix any contradictions in the current prompt (e.g., conflicting instructions about full country name vs ISO code).
   - Make the prompt more descriptive and specific to passports if it is currently too generic. For example, if the current prompt just says "Extract the fields from the document" you can make it more specific like "Extract the fields from the passport document. The document is a government-issued travel document that typically contains a photo, personal details, and official information. Use your knowledge of common passport formats and conventions to guide your extraction."
   - Do NOT remove any important details or constraints that are already in the prompt. Instead, only ADD or CLARIFY instructions to fix the identified issues. The refined prompt should be an improved version of the current prompt, not a simplified one.
   - Remove any dangling or empty bullet points.

4. Return ONLY the refined user prompt text. No explanations, no markdown fences, no commentary. The output should be the raw prompt text ready to be saved as user_prompt.txt."""


def call_refiner_llm(
    refiner_prompt: str,
    ocr_texts: list[str],
    page_images_b64: list[str],
) -> str:
    """Call GPT-4o with refiner meta-prompt + images + OCR, return refined user prompt."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env file")
    client = OpenAI(api_key=api_key)

    # Build multimodal user content: text prompt + images + OCR
    user_content: list[dict] = [
        {"type": "text", "text": refiner_prompt},
    ]
    for i, (text, img_b64) in enumerate(zip(ocr_texts, page_images_b64)):
        user_content.append({
            "type": "text",
            "text": f"--- Page {i + 1} OCR text ---\n{text}",
        })
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": REFINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise RuntimeError("GPT-4o returned empty response for prompt refinement")

    return raw.strip()


def main() -> None:
    # 1. Load prompts
    user_prompt = _load_user_prompt()
    print("=== Prompts Loaded ===")
    print(f"System prompt ({len(SYSTEM_PROMPT)} chars): {SYSTEM_PROMPT[:80]}...")
    print(f"User prompt ({len(user_prompt)} chars, {len(user_prompt.splitlines())} lines)")
    print()

    # 2. Load images + OCR cache — only for files that have errors
    files = find_input_files(INPUT_DIR)
    if not files:
        print(f"No supported files found in {INPUT_DIR}")
        return

    # Build lookup: filename stem -> file path
    file_map: dict[str, Path] = {f.stem: f for f in files}

    print(f"=== {len(files)} Source Files Available ===")
    print()

    # 3. Load comparison results
    print("=== Comparison Results ===")
    comparison_df, folder = load_comparison_results()
    print(f"Shape: {comparison_df.shape}")
    print()

    # 4. Build error report
    print("=== Building Error Report ===")
    error_report = build_error_report(comparison_df)
    for field_report in error_report:
        print(f"  {field_report['field']}: {field_report['error_count']} errors, {field_report['correct_count']} correct")

    # Save error report JSON
    error_report_dir = ERROR_REPORT_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    error_report_dir.mkdir(parents=True, exist_ok=True)
    error_report_path = error_report_dir / "error_report.json"
    error_report_path.write_text(
        json.dumps(error_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Error report saved to: {error_report_path}")
    print()

    # 5. Build refiner prompt
    print("=== Calling LLM for Prompt Refinement ===")
    refiner_prompt = build_refiner_prompt(user_prompt, error_report)
    print(f"Refiner prompt size: {len(refiner_prompt)} chars")
    print()

    # 6. Collect images + OCR: all error files + per-field correct samples
    error_files: set[str] = set()
    correct_files: set[str] = set()
    for field_report in error_report:
        for err in field_report["errors"]:
            error_files.add(err["source_file"])
        for ok in field_report["correct_samples"]:
            correct_files.add(ok["source_file"])

    # Per-field correct samples give better contrast than only "perfect" files.
    # A file wrong on first_name may still be correct on nationality — useful signal.
    files_to_send = sorted(error_files | correct_files)

    ocr_texts_to_send: list[str] = []
    images_to_send: list[str] = []
    for stem in files_to_send:
        if stem in file_map:
            pages_b64 = file_to_base64_images(file_map[stem], dpi=200, max_dim=1200)
            ocr_texts = ocr_pages(pages_b64, source_stem=stem)
            ocr_texts_to_send.extend(ocr_texts)
            images_to_send.extend(pages_b64)

    print(f"  Sending {len(error_files)} error files + {len(correct_files - error_files)} extra correct samples = {len(files_to_send)} files ({len(images_to_send)} pages) to LLM")
    print()

    # 7. Call LLM with images + OCR + error report
    refined_prompt = call_refiner_llm(refiner_prompt, ocr_texts_to_send, images_to_send)
    print(f"Refined prompt received: {len(refined_prompt)} chars, {len(refined_prompt.splitlines())} lines")
    print()

    # 8. Save
    output_path = folder / "refined_prompt.txt"
    output_path.write_text(refined_prompt, encoding="utf-8")
    print(f"Saved refined prompt to: {output_path}")


if __name__ == "__main__":
    main()
