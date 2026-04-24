"""
Find unused .tex files and images in LaTeX directory.
Traverses all Main_*.tex files and their \input{} inclusions,
collects all \includegraphics{} references, then identifies unused files.
"""

import os
import re
import shutil
from pathlib import Path

LATEX_DIR = Path(__file__).parent
IMAGES_DIR = LATEX_DIR / "images"
BACKUP_DIR = LATEX_DIR / "_backup_unused"

# Regex patterns
INPUT_PATTERN = re.compile(r'\\input\{([^}]+)\}')
INCLUDE_PATTERN = re.compile(r'\\include\{([^}]+)\}')
GRAPHICS_PATTERN = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')

used_tex_files = set()
used_images = set()

def normalize_tex_path(ref, base_dir):
    """Resolve a \input{} reference to an absolute path."""
    ref = ref.strip()
    if not ref.endswith('.tex'):
        ref = ref + '.tex'
    # Try relative to base_dir first, then to LATEX_DIR
    for search_dir in [base_dir, LATEX_DIR]:
        candidate = (search_dir / ref).resolve()
        if candidate.exists():
            return candidate
    return None

def normalize_image_path(ref):
    """Resolve an \includegraphics{} reference to a filename in images/."""
    ref = ref.strip()
    base = Path(ref).name  # just filename, no dir
    stem = Path(base).stem
    # Try with various extensions
    for ext in ['', '.pdf', '.png', '.jpg', '.jpeg', '.eps', '.PNG', '.JPG']:
        candidate = IMAGES_DIR / (stem + ext)
        if candidate.exists():
            return candidate
        # Also try the full ref name
        candidate2 = IMAGES_DIR / (ref + ext)
        if candidate2.exists():
            return candidate2
        # Try path as given relative to LATEX_DIR
        candidate3 = (LATEX_DIR / (ref + ext)).resolve()
        if candidate3.exists() and str(candidate3).startswith(str(IMAGES_DIR)):
            return candidate3
    # Try exact match
    candidate = IMAGES_DIR / base
    if candidate.exists():
        return candidate
    return None

def parse_file(tex_path):
    """Recursively parse a .tex file and collect all referenced files."""
    tex_path = Path(tex_path).resolve()
    if tex_path in used_tex_files:
        return
    if not tex_path.exists():
        return

    used_tex_files.add(tex_path)

    try:
        content = tex_path.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"  ERROR reading {tex_path}: {e}")
        return

    # Remove comments
    content_no_comments = re.sub(r'%.*', '', content)

    # Find \input{} and \include{} references
    for pattern in [INPUT_PATTERN, INCLUDE_PATTERN]:
        for match in pattern.finditer(content_no_comments):
            ref = match.group(1)
            resolved = normalize_tex_path(ref, tex_path.parent)
            if resolved:
                parse_file(resolved)
            else:
                print(f"  WARN: Could not resolve \\input{{{ref}}} in {tex_path.name}")

    # Find \includegraphics{} references
    for match in GRAPHICS_PATTERN.finditer(content_no_comments):
        ref = match.group(1)
        resolved = normalize_image_path(ref)
        if resolved:
            used_images.add(resolved.resolve())
        # else: image not found, might be generated or missing

def main():
    print("=" * 70)
    print("Scanning LaTeX directory for unused files")
    print("=" * 70)

    # Find all driver files: Main_*.tex + any .tex with \documentclass
    main_files = set(LATEX_DIR.glob("Main_*.tex"))
    for tex in LATEX_DIR.glob("*.tex"):
        try:
            content = tex.read_text(encoding='utf-8', errors='ignore')
            if r'\documentclass' in content:
                main_files.add(tex)
        except Exception:
            pass
    main_files = sorted(main_files)
    print(f"\nFound {len(main_files)} driver files (Main_*.tex + files with \\documentclass)")

    # Parse each Main file recursively
    print("\nParsing inclusions...")
    for mf in main_files:
        parse_file(mf)

    print(f"\nTotal referenced .tex files: {len(used_tex_files)}")
    print(f"Total referenced images: {len(used_images)}")

    # --- Find unused .tex files ---
    all_tex_files = set(p.resolve() for p in LATEX_DIR.glob("*.tex")
                        if p.name != "find_unused.py")
    # Also check images/tikz/*.tex
    tikz_tex = set(p.resolve() for p in (IMAGES_DIR / "tikz").glob("*.tex")) if (IMAGES_DIR / "tikz").exists() else set()

    unused_tex = all_tex_files - used_tex_files
    # Don't flag tikz files as unused (they're standalone)
    unused_tex_main = all_tex_files - tikz_tex - used_tex_files

    print(f"\nAll .tex files in LaTeX/: {len(all_tex_files)}")
    print(f"Unused .tex files: {len(unused_tex_main)}")

    # --- Find unused images ---
    all_images = set()
    for p in IMAGES_DIR.iterdir():
        if p.is_file():
            all_images.add(p.resolve())
    # Note: skipping images/tikz/ subdirectory files

    unused_images = all_images - used_images

    print(f"\nAll image files in LaTeX/images/: {len(all_images)}")
    print(f"Unused image files: {len(unused_images)}")

    # --- Report ---
    print("\n" + "=" * 70)
    print("UNUSED .TEX FILES:")
    print("=" * 70)
    for f in sorted(unused_tex_main):
        print(f"  {f.name}")

    print("\n" + "=" * 70)
    print("UNUSED IMAGE FILES (first 50 shown):")
    print("=" * 70)
    for f in sorted(unused_images)[:50]:
        print(f"  {f.name}")
    if len(unused_images) > 50:
        print(f"  ... and {len(unused_images) - 50} more")

    # --- Save full lists to files ---
    unused_tex_list = LATEX_DIR / "_unused_tex_files.txt"
    unused_img_list = LATEX_DIR / "_unused_image_files.txt"

    with open(unused_tex_list, 'w', encoding='utf-8') as f:
        for path in sorted(unused_tex_main):
            f.write(str(path) + '\n')

    with open(unused_img_list, 'w', encoding='utf-8') as f:
        for path in sorted(unused_images):
            f.write(str(path) + '\n')

    print(f"\nFull lists saved to:")
    print(f"  {unused_tex_list}")
    print(f"  {unused_img_list}")

    # --- Ask before moving ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY:")
    print(f"  Unused .tex files: {len(unused_tex_main)}")
    print(f"  Unused image files: {len(unused_images)}")
    print(f"  Backup destination: {BACKUP_DIR}")

    answer = input("\nMove unused files to backup? (yes/no): ").strip().lower()
    if answer != 'yes':
        print("Aborted. No files moved.")
        return

    # Create backup dirs
    backup_tex_dir = BACKUP_DIR / "tex"
    backup_img_dir = BACKUP_DIR / "images"
    backup_tex_dir.mkdir(parents=True, exist_ok=True)
    backup_img_dir.mkdir(parents=True, exist_ok=True)

    moved_tex = 0
    for f in sorted(unused_tex_main):
        dest = backup_tex_dir / f.name
        shutil.move(str(f), str(dest))
        moved_tex += 1

    moved_img = 0
    for f in sorted(unused_images):
        dest = backup_img_dir / f.name
        shutil.move(str(f), str(dest))
        moved_img += 1

    print(f"\nDone! Moved {moved_tex} .tex files and {moved_img} images to {BACKUP_DIR}")

if __name__ == '__main__':
    main()
