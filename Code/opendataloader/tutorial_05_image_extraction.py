"""
Tutorial 05 — Image Extraction Modes
Covers: image_output="off" | "embedded" | "external", image_format="png" | "jpeg".
Run:  python tutorial_05_image_extraction.py
"""

import os
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"

# NVIDIAAn.pdf likely contains figures/images
PDF = DATA_DIR / "NVIDIAAn.pdf"

modes = [
    ("off",      "png",  "Images are skipped entirely"),
    ("external", "png",  "Images saved as separate PNG files"),
    ("external", "jpeg", "Images saved as separate JPEG files"),
    ("embedded", "png",  "Images base64-encoded inside the Markdown"),
]

for img_output, img_fmt, description in modes:
    out_subdir = Path(__file__).parent / "output" / f"05_images_{img_output}_{img_fmt}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Mode: image_output={img_output!r}  image_format={img_fmt!r}")
    print(f"  {description}")

    opendataloader_pdf.convert(
        input_path=[str(PDF)],
        output_dir=str(out_subdir),
        format="markdown",
        image_output=img_output,
        image_format=img_fmt,
    )

    # Count output files
    all_files = list(out_subdir.rglob("*"))
    md_files  = [f for f in all_files if f.suffix == ".md"]
    img_files = [f for f in all_files if f.suffix in {".png", ".jpg", ".jpeg"}]

    print(f"  Markdown files : {len(md_files)}")
    print(f"  Image files    : {len(img_files)}")

    # Show image filenames if external mode
    for img in img_files[:5]:
        size = img.stat().st_size
        print(f"    {img.name}  ({size:,} bytes)")
    if len(img_files) > 5:
        print(f"    ... and {len(img_files)-5} more")

    # For embedded mode, show size of the Markdown (images inflate it)
    if img_output == "embedded" and md_files:
        md_size = md_files[0].stat().st_size
        print(f"  Markdown size  : {md_size:,} bytes  (includes base64 images)")

    print()

print("Summary: 'external' saves images alongside the document.")
print("         'embedded' keeps everything in one Markdown file.")
print("         'off' is fastest when you only need text.")
