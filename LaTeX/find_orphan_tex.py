"""
Find .tex files under LaTeX/ that are not reachable from any Main_*.tex driver
via \\input{}/\\include{} (transitively). This is the reverse check from the
latex-audit skill's "inputs" mode (which flags \\input targets missing from
disk, not files on disk missing from every chain).

Skips backup/_backup/_retired directories, matching repo convention. Also
excludes anything under images/: those .tex files are standalone TikZ/PGF
sources compiled independently to PDF (see CLAUDE.md), not part of any
Main_*.tex \\input chain, so they'd otherwise be reported as false positives.

Usage: python find_orphan_tex.py [path-to-LaTeX-dir]
"""
import re
import sys
from pathlib import Path

LATEX_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent).resolve()

EXCLUDED_DIR_NAMES = {'backup', '_backup', '_retired'}
INPUT_RE = re.compile(r'\\(input|include)\{([^}]+)\}')
COMMENT_LINE_RE = re.compile(r'^\s*%')


def is_excluded(path: Path) -> bool:
    names = {part.lower() for part in path.parts}
    return bool(names & EXCLUDED_DIR_NAMES)


def is_image_source(path: Path) -> bool:
    return any(part.lower() == 'images' for part in path.parts)


def get_input_targets(file: Path):
    targets = []
    try:
        text = file.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"WARN: could not read {file}: {e}", file=sys.stderr)
        return targets
    for line in text.splitlines():
        if COMMENT_LINE_RE.match(line):
            continue
        for m in INPUT_RE.finditer(line):
            targets.append(m.group(2))
    return targets


def resolve_chain(start: Path, all_visited: set, missing: list):
    queue = [start.resolve()]
    local_visited = set()
    while queue:
        current = queue.pop(0)
        if current in local_visited:
            continue
        local_visited.add(current)
        all_visited.add(current)
        if not current.exists():
            continue
        directory = current.parent
        for name in get_input_targets(current):
            target_name = name if name.endswith('.tex') else name + '.tex'
            candidate = directory / target_name
            if candidate.exists():
                resolved = candidate.resolve()
                if resolved not in local_visited:
                    queue.append(resolved)
            else:
                missing.append((str(start), str(current), name, str(candidate)))
    return local_visited


def main():
    all_tex = [p for p in LATEX_DIR.rglob('*.tex') if not is_excluded(p)]
    all_tex_resolved = {p.resolve() for p in all_tex}

    drivers = sorted(p for p in all_tex if p.name.startswith('Main_'))

    reachable = set()
    missing_all = []
    for d in drivers:
        resolve_chain(d, reachable, missing_all)

    orphan_candidates = all_tex_resolved - reachable
    orphans = sorted(p for p in orphan_candidates if not is_image_source(p))
    image_false_positives = sorted(p for p in orphan_candidates if is_image_source(p))

    print(f"Total .tex files under {LATEX_DIR}: {len(all_tex_resolved)}")
    print(f"Driver files (Main_*.tex): {len(drivers)}")
    print(f"Reachable via input/include chains (incl. drivers): {len(reachable)}")
    print(f"Excluded as standalone image sources (images/*.tex): {len(image_false_positives)}")
    print(f"Orphan files (on disk, not reachable from any driver): {len(orphans)}")
    print()
    if orphans:
        print("--- ORPHAN FILES ---")
        for o in orphans:
            rel = o.relative_to(LATEX_DIR.resolve())
            print(f"  {rel}")
    print()
    if missing_all:
        print(f"--- (side note) {len(missing_all)} unresolved \\input targets found while walking chains ---")
        for driver, frm, target, expected in missing_all[:20]:
            print(f"  driver={Path(driver).name} from={Path(frm).name} target={target} expected={expected}")
        if len(missing_all) > 20:
            print(f"  ... and {len(missing_all) - 20} more")


if __name__ == '__main__':
    main()
