"""
Tutorial 06 — Batch Processing
Covers: converting an entire folder of PDFs in one call, per-file timing,
        and a summary table of results.
Run:  python tutorial_06_batch_processing.py
"""

import time
import json
from pathlib import Path
from collections import Counter
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "06_batch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pdfs = sorted(DATA_DIR.glob("*.pdf"))
print(f"Found {len(pdfs)} PDF(s) in {DATA_DIR}\n")

# ── Option A: pass the whole folder ──────────────────────────────────────────
print("Option A — convert entire folder at once")
t0 = time.perf_counter()
opendataloader_pdf.convert(
    input_path=[str(DATA_DIR)],
    output_dir=str(OUT_DIR / "folder_mode"),
    format="json,markdown",
)
folder_elapsed = time.perf_counter() - t0
print(f"  Folder mode finished in {folder_elapsed:.3f}s")

# ── Option B: pass a list of files ────────────────────────────────────────────
print("\nOption B — convert explicit file list")
t0 = time.perf_counter()
opendataloader_pdf.convert(
    input_path=[str(p) for p in pdfs],
    output_dir=str(OUT_DIR / "list_mode"),
    format="json,markdown",
)
list_elapsed = time.perf_counter() - t0
print(f"  List mode finished in {list_elapsed:.3f}s")

# ── Per-file stats from JSON outputs ─────────────────────────────────────────
print("\n\nPer-file element breakdown (from list_mode JSON outputs)")
print(f"{'File':45s} {'Elements':>8s} {'Pages':>6s} {'Types'}")
print("─" * 80)

list_out = OUT_DIR / "list_mode"
for pdf in pdfs:
    json_file = list_out / (pdf.stem + ".json")
    if not json_file.exists():
        print(f"  {pdf.name:43s}  (no JSON output)")
        continue
    elements = json.loads(json_file.read_text(encoding="utf-8"))
    pages = {el.get("page number") for el in elements if el.get("page number")}
    types = Counter(el.get("type", "?") for el in elements)
    top_types = ", ".join(f"{t}:{n}" for t, n in types.most_common(3))
    print(f"  {pdf.name:43s}  {len(elements):8d}  {len(pages):6d}  {top_types}")

print(f"\nFolder mode: {folder_elapsed:.3f}s   List mode: {list_elapsed:.3f}s")
