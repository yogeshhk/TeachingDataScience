# Contributing

Thanks for considering a contribution — corrections, new content, code examples, and citation
fixes are all welcome. This repo runs informally: no CLA, no heavyweight process, just a few
conventions worth following so new material fits in cleanly.

## Ways to contribute

- **Fix something**: a typo, a broken code example, a wrong formula, a dead link, a missing
  citation. Small, focused pull requests are easiest to review and merge.
- **Add a new topic slide**: see below.
- **Add or improve a `Code/` example**: runnable scripts/notebooks that pair with a `LaTeX/`
  topic are especially valuable — see [`Code/README.md`](Code/README.md) for how the directory
  is organized.
- **Report an issue**: even if you don't have time to fix it yourself, flagging a bug, a stale
  reference, or a missing citation is useful.

## Adding a new LaTeX topic

1. Create `LaTeX/<domain>_<topic>.tex` with Beamer frames (see naming convention below).
2. Copy [`Main_Sample_Presentation.tex`](LaTeX/Main_Sample_Presentation.tex) /
   [`Main_Sample_CheatSheet.tex`](LaTeX/Main_Sample_CheatSheet.tex) (both drive
   [`sample_content.tex`](LaTeX/sample_content.tex)) as a quick reference for the common frame
   patterns already used throughout the repo: a section-heading frame, a list slide, an image,
   a code listing, a two-column layout, and a table.
3. `\input{<domain>_<topic>}` inside the relevant `seminar_*_content.tex` (or create a new one —
   see [`COURSES.md`](COURSES.md) for the existing catalog, so you can tell whether your topic
   fits an existing seminar or needs a new one).
4. Place supporting images in `LaTeX/images/`.
5. Compile before submitting — from `LaTeX/`:
   ```
   texify -cp Main_Seminar_<YourSeminar>_Presentation.tex
   ```
   A clean compile (no errors, images render) is the main thing reviewers will check.

### Naming conventions

- Topic files: `<domain>_<topic>.tex` (e.g. `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Driver files: `Main_[Course|Workshop|Seminar]_<Subject>_[Presentation|CheatSheet].tex`
- Content hierarchy: **Course** (~1-2 weeks) → **Workshop** (~1-2 days) → **Seminar** (~1-2 hrs)
  → raw topic files — see the README for what each tier means
- Every Seminar and Workshop should have both a `_Presentation.tex` (Beamer slides) and a
  `_CheatSheet.tex` (two-column notes) driver, sharing the same content file

### Style notes

- Match the existing frame boilerplate (see `sample_content.tex`) rather than introducing a new
  slide structure or theme.
- Cite sources. Most content here is built by learning from public material — a `{\tiny (Ref:
  ...)}` line under the frame title is the repo's existing convention for that.
- Keep code examples runnable and current — a broken `pip install` or a removed API (e.g. a
  library function deprecated since the slide was written) is exactly the kind of thing worth
  fixing even in existing content.

## Adding or improving `Code/`

- Match a `Code/<subdir>/` name to its corresponding `LaTeX/` topic where one exists.
- Use a `conda`-based `environment.yml` for dependencies (not `pip`/`venv`) — see existing
  subdirectories for the pattern.
- If you add a `test_*.py`, it should run cleanly with `pytest` and avoid real API calls or
  large model downloads (mock them, same as the existing test suites).

## Pull requests

- Keep PRs focused — one topic, one fix, or one small cluster of related changes.
- Describe what changed and why in the PR description; for content changes, note what you
  verified (e.g. "compiled clean, N pages, images render").
- No strict formatting/linting gate — just try to match the surrounding file's existing style.

## Questions

Open an issue if anything above is unclear, or if you're not sure whether something you want to
add fits the repo's scope.
