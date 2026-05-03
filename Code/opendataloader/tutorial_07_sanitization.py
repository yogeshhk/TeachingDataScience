"""
Tutorial 07 — AI Safety & Sanitization
Covers: sanitize=True (emails, URLs, phones → placeholders), hidden-text
        detection, and comparing sanitized vs raw output.
Run:  python tutorial_07_sanitization.py
"""

import re
from pathlib import Path
import opendataloader_pdf

DATA_DIR = Path(__file__).parent / "data"
OUT_RAW  = Path(__file__).parent / "output" / "07_sanitize" / "raw"
OUT_SAFE = Path(__file__).parent / "output" / "07_sanitize" / "sanitized"
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_SAFE.mkdir(parents=True, exist_ok=True)

# NVIDIAAn.pdf is a corporate document — likely contains URLs / contact info
PDF = DATA_DIR / "NVIDIAAn.pdf"

# ── Raw extraction ────────────────────────────────────────────────────────────
print("Extracting WITHOUT sanitization ...")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_RAW),
    format="text",
    sanitize=False,
)

# ── Sanitized extraction ──────────────────────────────────────────────────────
print("Extracting WITH sanitization ...")
opendataloader_pdf.convert(
    input_path=[str(PDF)],
    output_dir=str(OUT_SAFE),
    format="text",
    sanitize=True,
)

# ── Compare ───────────────────────────────────────────────────────────────────
raw_file  = OUT_RAW  / (PDF.stem + ".txt")
safe_file = OUT_SAFE / (PDF.stem + ".txt")

if not raw_file.exists() or not safe_file.exists():
    print("Output files not found — check format/stem matching.")
else:
    raw_text  = raw_file.read_text(encoding="utf-8")
    safe_text = safe_file.read_text(encoding="utf-8")

    # Simple regex patterns to hunt for sensitive data
    email_re = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    url_re   = re.compile(r"https?://\S+")
    phone_re = re.compile(r"\+?[\d][\d\s\-().]{7,}\d")

    def report(label, text):
        emails = email_re.findall(text)
        urls   = url_re.findall(text)
        phones = phone_re.findall(text)
        print(f"\n  [{label}]")
        print(f"    Emails : {len(emails)}")
        print(f"    URLs   : {len(urls)}")
        print(f"    Phones : {len(phones)}")
        if emails:
            print(f"    Sample email : {emails[0]}")
        if urls:
            print(f"    Sample URL   : {urls[0][:60]}")

    print("\n" + "=" * 60)
    print("Sensitive-data comparison")
    print("=" * 60)
    report("Raw (sanitize=False)", raw_text)
    report("Safe (sanitize=True)", safe_text)

    # Show a snippet where they differ
    raw_lines  = raw_text.splitlines()
    safe_lines = safe_text.splitlines()
    diffs = [(r, s) for r, s in zip(raw_lines, safe_lines) if r != s]
    if diffs:
        print(f"\nFirst differing line (raw vs sanitized):")
        print(f"  RAW : {diffs[0][0][:100]}")
        print(f"  SAFE: {diffs[0][1][:100]}")
    else:
        print("\nNo line-level differences found (document may not contain "
              "recognisable PII patterns).")
