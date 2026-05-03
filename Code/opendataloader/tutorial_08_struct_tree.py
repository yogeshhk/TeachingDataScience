"""
Tutorial 08 — Native PDF Structure Tags (use_struct_tree)
Covers: extracting the logical structure tree embedded in tagged PDFs,
        comparing struct-tree output with the default heuristic output,
        and generating a tagged-PDF for accessibility.
Run:  python tutorial_08_struct_tree.py
"""

import json
from collections import Counter
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR  = Path(__file__).parent / "output" / "08_struct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Systems Thinking docs are well-structured and likely tagged
PDF = DATA_DIR / "Systems Thinking in 25 Words or Less.pdf"

# ── Heuristic mode (default) ─────────────────────────────────────────────────
print("Mode A — heuristic (use_struct_tree=False)")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR / "heuristic"),
    format="json",
    use_struct_tree=False,
)

# ── Struct-tree mode ──────────────────────────────────────────────────────────
print("Mode B — native struct tree (use_struct_tree=True)")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR / "struct_tree"),
    format="json",
    use_struct_tree=True,
)

# ── Compare element-type distributions ───────────────────────────────────────
def load_elements(base):
    p = base / (PDF.stem + ".json")
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))

heuristic_els = load_elements(OUT_DIR / "heuristic")
struct_els    = load_elements(OUT_DIR / "struct_tree")

def type_counts(els):
    return Counter(el.get("type", "?") for el in els)

print("\n" + "=" * 60)
print(f"{'Type':20s} {'Heuristic':>10s} {'StructTree':>10s}")
print("─" * 42)

all_types = sorted(type_counts(heuristic_els).keys() |
                   type_counts(struct_els).keys())
h_counts = type_counts(heuristic_els)
s_counts = type_counts(struct_els)

for t in all_types:
    print(f"  {t:18s} {h_counts.get(t,0):10d} {s_counts.get(t,0):10d}")

print(f"\n  {'TOTAL':18s} {len(heuristic_els):10d} {len(struct_els):10d}")

# ── Tagged-PDF output (accessibility export) ──────────────────────────────────
print("\n" + "=" * 60)
print("Generating tagged-PDF for accessibility ...")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_DIR / "tagged_pdf"),
    format="tagged-pdf",
)

tagged_out = OUT_DIR / "tagged_pdf"
tagged_files = list(tagged_out.glob("*.pdf"))
if tagged_files:
    for f in tagged_files:
        print(f"  Tagged PDF: {f.name}  ({f.stat().st_size:,} bytes)")
else:
    print("  (No tagged PDF found — feature may require additional setup)")
