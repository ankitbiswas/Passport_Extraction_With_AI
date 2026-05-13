# Passport Data Extraction

LLM-powered pipeline that extracts structured data from passport documents (PDFs/images) using GPT-4o with OCR fallback.

## Architecture

```
input_pdfs/          ← Drop passport PDFs/images here
    ↓
pdf_processor.py     ← PDF → base64 PNG pages (with resize for API limits)
    ↓
ocr.py               ← pytesseract OCR with SHA256 disk cache
    ↓
llm_extractor.py     ← GPT-4o structured extraction (JSON schema mode)
    ↓
output/              ← Per-file JSON results
    ↓
analysis.py          ← Compare extracted vs ground truth, compute metrics
    ↓
prompt_refiner.py    ← Feed errors + images back to LLM → refined prompt
    ↓
user_prompt.txt      ← Extraction prompt (auto-improved by refiner)
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point — batch extract from `input_pdfs/` |
| `llm_extractor.py` | GPT-4o call with structured JSON output |
| `pdf_processor.py` | PDF/image → base64 PNG, auto-resize for API limits |
| `ocr.py` | pytesseract OCR with SHA256 disk cache |
| `schema.py` | Pydantic model: 8 passport fields |
| `analysis.py` | Compare extracted vs `GT.xlsx`, output metrics + comparison |
| `prompt_refiner.py` | Analyze errors, send to LLM, produce refined `user_prompt.txt` |
| `user_prompt.txt` | Current extraction prompt (updated by refiner) |

## Extracted Fields

`first_name`, `last_name`, `passport_number`, `nationality`, `date_of_birth`, `date_of_expiry`, `issuing_country`, `issuing_authority`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`.

Create `.env`:
```
OPENAI_API_KEY=sk-...
```

## Usage

### 1. Extract passport data

Drop PDFs/images into `input_pdfs/`, then:

```powershell
python main.py
```

Results land in `output/<timestamp>/` as individual JSON files.

### 2. Compare against ground truth

Place `GT.xlsx` in project root with columns: `source_file`, `first_name`, `last_name`, `passport_number`, `nationality`, `date_of_birth`, `date_of_expiry`, `issuing_country`, `issuing_authority`.

```powershell
python analysis.py
```

Outputs `comparison_results/<timestamp>/`:
- `comparison_results.xlsx` — side-by-side GT vs extracted with TP/FP/FN/MISTAKE labels
- `metrics.xlsx` — per-field accuracy, precision, recall, F1

### 3. Refine the extraction prompt

After running analysis:

```powershell
python prompt_refiner.py
```

Sends error cases + correct samples (images + OCR) to GPT-4o. The LLM analyzes failure patterns and rewrites `user_prompt.txt`. Refined prompt saved to `comparison_results/<timestamp>/refined_prompt.txt`.

Copy it to `user_prompt.txt` to apply, then re-run extraction to measure improvement.

## Key Dependencies

- `openai` — GPT-4o API
- `pytesseract` — OCR text extraction
- `pdf2image` / `Pillow` — PDF rendering, image processing
- `pandas` / `openpyxl` — Excel I/O for GT and comparison
- `pydantic` — Schema definition and JSON validation
