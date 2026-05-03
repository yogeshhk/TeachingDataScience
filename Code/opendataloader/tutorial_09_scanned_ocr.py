"""
Tutorial 09 — Scanned PDF & OCR via Hybrid Mode
Covers: detecting image-only pages, hybrid-mode setup, OCR language options.

Prerequisites — start the hybrid backend in a SEPARATE terminal first:
    conda activate opendataloader
    opendataloader-pdf-hybrid --port 5002 --force-ocr

Then run this script:
    python tutorial_09_scanned_ocr.py
"""

import json
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "09_ocr"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF = DATA_DIR / "scanned.pdf"

# ── Step 1: attempt local-mode extraction (no OCR) ───────────────────────────
print("Step 1 — local mode (no OCR)")
print("      Image-only pages will produce empty text elements.\n")

opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR / "local"),
    format="json,markdown",
)

local_json = OUT_DIR / "local" / (PDF.stem + ".json")
if local_json.exists():
    elements = json.loads(local_json.read_text(encoding="utf-8"))
    non_empty = [el for el in elements
                 if el.get("content", "").strip()]
    print(f"  Total elements : {len(elements)}")
    print(f"  Non-empty text : {len(non_empty)}")
    if len(non_empty) == 0:
        print("  >> This is a fully scanned PDF — OCR is required.")
    else:
        print("  Sample text:", non_empty[0].get("content","")[:120])

# ── Step 2: hybrid mode with OCR ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2 — hybrid mode (OCR)")
print("  Requires: opendataloader-pdf-hybrid --port 5002 --force-ocr")
print("=" * 60)

try:
    opendataloader_pdf.convert(
        input_path=[str(PDF)],
        output_dir=str(OUT_DIR / "ocr"),
        format="json,markdown",
        hybrid="docling-fast",
    )

    ocr_json = OUT_DIR / "ocr" / (PDF.stem + ".json")
    if ocr_json.exists():
        elements = json.loads(ocr_json.read_text(encoding="utf-8"))
        non_empty = [el for el in elements
                     if el.get("content", "").strip()]
        print(f"\n  Total elements : {len(elements)}")
        print(f"  Non-empty text : {len(non_empty)}")
        if non_empty:
            print("\n  First extracted text:")
            for el in non_empty[:3]:
                print(f"    [{el.get('type','?')}] {el.get('content','')[:100]}")

except Exception as exc:
    print(f"\n  Hybrid backend not reachable: {exc}")
    print("\n  To enable OCR, open a second terminal and run:")
    print("    conda activate opendataloader")
    print("    opendataloader-pdf-hybrid --port 5002 --force-ocr")
    print("\n  For non-English documents add --ocr-lang, e.g.:")
    print("    opendataloader-pdf-hybrid --port 5002 --force-ocr --ocr-lang 'hi,en'")

# ── Multi-language OCR reference ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Multi-language OCR language codes (sample)")
print("─" * 40)
LANG_CODES = {
    "en": "English", "hi": "Hindi", "de": "German",
    "fr": "French",  "ar": "Arabic", "ko": "Korean",
    "ja": "Japanese", "ch_sim": "Chinese Simplified",
    "ch_tra": "Chinese Traditional",
}
for code, name in LANG_CODES.items():
    print(f"  {code:8s}  {name}")
print("\nPass as comma-separated string: --ocr-lang 'en,hi,de'")
