"""
Tutorial 02 — Output Formats
Covers: markdown, json, html, text — all produced in one call.
Run:  python tutorial_02_output_formats.py
"""

import time
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "02_formats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF = DATA_DIR / "Systems Thinking in 25 Words or Less.pdf"

FORMATS = "markdown,json,html,text"

print("=" * 60)
print(f"PDF : {PDF.name}")
print(f"Formats requested: {FORMATS}")
print("=" * 60)

t0 = time.perf_counter()
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR),
    format=FORMATS,
)
elapsed = time.perf_counter() - t0
print(f"Conversion finished in {elapsed:.3f}s\n")

# ── Report what was produced ─────────────────────────────────────────────────
extensions = {".md": "Markdown", ".json": "JSON", ".html": "HTML", ".txt": "Text"}
for ext, label in extensions.items():
    path = OUT_DIR / (PDF.stem + ext)
    if path.exists():
        size = path.stat().st_size
        print(f"  [{label:8s}]  {path.name}  ({size:,} bytes)")
    else:
        print(f"  [{label:8s}]  NOT FOUND")

# ── Preview each format ──────────────────────────────────────────────────────
PREVIEW = 400

for ext, label in extensions.items():
    path = OUT_DIR / (PDF.stem + ext)
    if not path.exists():
        continue
    print(f"\n{'─' * 60}")
    print(f"{label} preview ({PREVIEW} chars)")
    print('─' * 60)
    try:
        text = path.read_text(encoding="utf-8")
        print(text[:PREVIEW])
    except Exception as exc:
        print(f"Could not read: {exc}")
