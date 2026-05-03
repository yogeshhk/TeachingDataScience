"""
Tutorial 04 — Table Extraction
Covers: isolating table elements from JSON output, rendering table content,
        and comparing Markdown vs JSON table representation.
Run:  python tutorial_04_table_extraction.py
"""

import json
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "04_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF = DATA_DIR / "table.pdf"

# ── Convert to both formats so we can compare ────────────────────────────────
print(f"Converting: {PDF.name}")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR),
    format="json,markdown",
)

# ── JSON: inspect table elements ──────────────────────────────────────────────
json_file = OUT_DIR / (PDF.stem + ".json")
elements  = json.loads(json_file.read_text(encoding="utf-8"))

tables = [el for el in elements if el.get("type") == "table"]
print(f"\nTables found in JSON: {len(tables)}")

for i, tbl in enumerate(tables, 1):
    page = tbl.get("page number", "?")
    bb   = tbl.get("bounding box", [])
    text = tbl.get("content", "")
    print(f"\n  Table {i}  (page {page})")
    if bb:
        print(f"  Bounding box: left={bb[0]:.1f} bottom={bb[1]:.1f} "
              f"right={bb[2]:.1f} top={bb[3]:.1f}")
    print(f"  Content preview:\n{text[:400]}")

# ── Markdown: show the rendered table block(s) ────────────────────────────────
md_file = OUT_DIR / (PDF.stem + ".md")
if md_file.exists():
    md = md_file.read_text(encoding="utf-8")

    # Locate lines that look like Markdown table rows (contain |)
    table_lines = [ln for ln in md.splitlines() if "|" in ln]
    if table_lines:
        print(f"\n\nMarkdown table rows ({len(table_lines)} lines):")
        print("─" * 60)
        for ln in table_lines[:30]:
            print(ln)
    else:
        print("\nNo Markdown table rows detected — trying full Markdown preview:")
        print(md[:600])

# ── Summary ───────────────────────────────────────────────────────────────────
non_table = [el for el in elements if el.get("type") != "table"]
print(f"\n\nAll elements in {PDF.name}")
print(f"  Tables     : {len(tables)}")
print(f"  Non-tables : {len(non_table)}")
