# CLAUDE.md

Open-source Data Science teaching material: `LaTeX/` (Beamer decks and two-column cheat sheets) and
`Code/` (Python scripts and notebooks). Read [`AUTHORING.md`](AUTHORING.md) before authoring, compiling
or testing; every how-to (build, content hierarchy, naming, compile policy, `Code/`, tests) lives only there.

## Never
- **No em dashes** in `.tex` or user-visible `.md` files. The `PreToolUse` hook
  `.claude/hooks/check-em-dash.ps1` blocks them (details in `AUTHORING.md`).
- **Never commit compiled PDFs or build litter**; the repo is public.
- **No mega-deck compile**: never run `make_all.bat` or the combined `Main_Course_MLCoEP_*` driver;
  ask before any compile other than the per-session MLCoEP drivers.
- **Conda only** for Python environments (`environment.yml`); no `venv` or `.venv`.
- MLCoEP PDF destinations are set by `D:\Yogesh\GitHub\CLAUDE.md`.
