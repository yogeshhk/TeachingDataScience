"""
Tutorial 03 — JSON Structure Analysis
Covers: parsing JSON output, counting element types, exploring heading
        hierarchy, and inspecting bounding boxes.
Run:  python tutorial_03_json_structure.py
"""

import json
from collections import Counter
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "03_json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF = DATA_DIR / "Systems Thinking Four Key Questions.pdf"

# ── Convert to JSON ──────────────────────────────────────────────────────────
print(f"Converting: {PDF.name}")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR),
    format="json",
)

json_file = OUT_DIR / (PDF.stem + ".json")
if not json_file.exists():
    raise FileNotFoundError(f"Expected {json_file}")

elements = json.loads(json_file.read_text(encoding="utf-8"))
print(f"Total elements: {len(elements)}\n")

# ── Element-type breakdown ────────────────────────────────────────────────────
counts = Counter(el.get("type", "unknown") for el in elements)
print("Element types")
print("─" * 40)
for etype, n in counts.most_common():
    print(f"  {etype:20s} {n:4d}")

# ── Heading hierarchy ─────────────────────────────────────────────────────────
headings = [el for el in elements if el.get("type") == "heading"]
if headings:
    print(f"\nHeadings ({len(headings)} total)")
    print("─" * 40)
    for h in headings:
        level  = h.get("heading level", 1)
        indent = "  " * (level - 1)
        text   = h.get("content", "").replace("\n", " ")[:70]
        print(f"  H{level} {indent}{text}")

# ── Page distribution ─────────────────────────────────────────────────────────
pages = Counter(el.get("page number") for el in elements if el.get("page number"))
print(f"\nElements per page")
print("─" * 40)
for page in sorted(pages):
    bar = "#" * pages[page]
    print(f"  Page {page:3d}: {bar} ({pages[page]})")

# ── Bounding-box sample ───────────────────────────────────────────────────────
print(f"\nBounding-box sample (first 5 elements)")
print("─" * 60)
print(f"  {'type':15s} {'page':5s} {'left':7s} {'bottom':7s} {'right':7s} {'top':7s}")
for el in elements[:5]:
    bb    = el.get("bounding box", [0, 0, 0, 0])
    etype = el.get("type", "?")
    page  = el.get("page number", "?")
    print(f"  {etype:15s} {str(page):5s} "
          f"{bb[0]:7.1f} {bb[1]:7.1f} {bb[2]:7.1f} {bb[3]:7.1f}")

# ── Font info sample ──────────────────────────────────────────────────────────
fonts = [(el.get("font","?"), el.get("font size", 0))
         for el in elements if el.get("font")]
if fonts:
    unique_fonts = sorted({(f, s) for f, s in fonts}, key=lambda x: -x[1])
    print(f"\nDistinct fonts (sorted by size)")
    print("─" * 40)
    for font, size in unique_fonts[:10]:
        print(f"  {size:5.1f}pt  {font}")
