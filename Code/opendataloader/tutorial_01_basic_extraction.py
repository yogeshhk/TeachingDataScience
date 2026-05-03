"""
Tutorial 01 — Basic Extraction
Covers: single-file conversion to Markdown and plain text.
Run:  python tutorial_01_basic_extraction.py
"""

import time
from pathlib import Path
import opendataloader_pdf

DATA_DIR  = Path(__file__).parent / "data"
OUT_DIR   = Path(__file__).parent / "output" / "01_basic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF = DATA_DIR / "sample.pdf"

# ── Markdown extraction ──────────────────────────────────────────────────────
print("=" * 60)
print("Extracting to Markdown ...")
t0 = time.perf_counter()

opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR),
    format="markdown",
)

elapsed = time.perf_counter() - t0
md_file = OUT_DIR / (PDF.stem + ".md")
print(f"Done in {elapsed:.3f}s  →  {md_file}")

if md_file.exists():
    content = md_file.read_text(encoding="utf-8")
    print(f"Characters: {len(content)}")
    print("\n--- First 800 characters ---")
    print(content[:800])

# ── Plain-text extraction ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Extracting to plain text ...")
t0 = time.perf_counter()

opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR),
    format="text",
)

elapsed = time.perf_counter() - t0
txt_file = OUT_DIR / (PDF.stem + ".txt")
print(f"Done in {elapsed:.3f}s  →  {txt_file}")

if txt_file.exists():
    content = txt_file.read_text(encoding="utf-8")
    print(f"Characters: {len(content)}")
    print("\n--- First 800 characters ---")
    print(content[:800])
