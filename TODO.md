# TODO

Working checklist from the 2026-09-13 repo review. Items are worked one at a time, checked off
here as each completes, per the repo's confirm-before-executing convention. This file is the
live tracker for repo-wide items; sub-project trackers are linked at the bottom rather than
duplicated here.

## Discoverability & first impressions

- [x] Fix the README's broken "See It In Action" images: `docs/screenshots/` is missing from
      disk, so all three sample-slide images 404. Restore or regenerate the three PNGs
      (`ml-ensemble.png`, `llm-embeddings.png`, `python-basics.png`).
      Done 2026-09-13: the deletion was deliberate (user removed `docs/` since it didn't follow
      repo naming/location convention), then the user restored the three original PNGs. Verified
      all three still show real, matching slide content (Ensemble Constructing, Tokenisation &
      Token Embedding, Compiled Languages) with no rendering defects. Moved to
      `LaTeX/images/screenshots/` per the user's preference, `docs/` removed entirely, and
      `README.md`'s three `<img src>` paths updated to match. (Along the way, briefly started
      compiling three decks to regenerate fresh screenshots instead; stopped two of the three
      background compiles once the user said the originals were restored, and cleaned up the one
      leftover PDF from the compile that had already finished.)

## Documentation hygiene

- [x] Rewrite or delete `SECURITY.md`: currently the untouched GitHub default template
      (references nonsensical "Supported Versions 5.1.x/5.0.x/4.0.x" for a slide-deck repo).
      Done 2026-09-13: rewritten with content scoped to this repo (leaked credentials, insecure
      example code), pointing to `CONTRIBUTING.md` for everything else.

## LaTeX structure & naming

- [x] Rename `MMMCoE_Seminar_GenAI_Presentation.tex` to follow the `Main_Seminar_*` naming
      convention, so it's visible to the course catalog and to the audit tooling.
      Done 2026-09-13: renamed to `Main_Seminar_GenAI_MMMCoE_Presentation.tex` (only other
      reference found was this TODO itself); added to `COURSES.md` under a new "Venue-specific
      talks" bullet.
- [x] Archive the ~33 clearly-superseded draft `.tex` files (names containing `_old`, `_v0`,
      `_v1`, `_v2`, `_del`, `_depricated`, `_bk`, `_tooold`) into `LaTeX/_retired/`. Do not
      delete; move only.
      Done 2026-09-13: spot-checked first via grep across the whole repo; all 34 (Bucket A of
      `orphan_tex_report.md`) turned out to already be explicitly commented-out `\input` lines,
      each annotated "(archived)"/"(deprecated)"/"(older...)" in their owning
      `seminar_*_content.tex`/`workshop_*_content.tex` files, so none were live anywhere. Moved
      to a new `LaTeX/_retired/superseded_drafts/` folder (matching the existing themed-subfolder
      convention under `_retired/`). Verified after the move: orphan count dropped 109 -> 75, and
      `latex-audit -Mode inputs` reports 0 of 279 drivers blocked (the commented-out references
      were never live, so nothing broke).

## Orphan-file analysis

- [x] Write the full 125-file orphan `.tex` report (from `find_orphan_tex.py`) to a file, grouped
      into: false positives (`images/*.tex` TikZ sources), superseded drafts (see above),
      possible `_short` sibling drift, and real content gaps (~55 files, e.g. the `graph_rag_impl_*`
      cluster, the Rasa chatbot example set, the `ai_tools_claudecode_*`/`notebooklm`/`sarvam`
      cluster, `seminar_artificialintelligence_tools_content.tex` with no driver at all) for
      review over future sessions.
      Done 2026-09-13: written to `LaTeX/orphan_tex_report.md`. Current count (after the MMMCoE
      rename added one more driver): 109 orphans + 15 image-source false positives. Bucket A
      (34 superseded drafts) feeds directly into item 5 below. Bucket B (17 possible `_short`
      sibling-drift files) and Bucket C (58 real content gaps, grouped into 15 named clusters,
      e.g. a near-complete "AI tools" mini-course and a `graph_rag_impl_*` cluster) are left as a
      decision list for future sessions, not resolved here.
- [x] Save `find_orphan_tex.py` permanently in the repo (candidate locations:
      `Code/claudecode/dot_claude/skills/latex-audit/`, alongside the existing PowerShell
      `latex-audit.ps1`, or directly under `LaTeX/`). Exclude `LaTeX/images/` from the scan so it
      stops false-flagging standalone TikZ sources.
      Done 2026-09-13: saved as `LaTeX/find_orphan_tex.py`, not the skills folder (that repo copy
      under `Code/claudecode/dot_claude/skills/latex-audit/` is a static teaching mirror of the
      real skill at `C:\Users\yoges\.claude\skills\latex-audit\`, which is shared across other
      repos, so extending it wasn't in scope here). Excludes `images/*.tex` from the orphan list
      and reports them separately as accounted-for. Also fixed a path-matching bug found while
      wiring this up: `_retired`/`backup` exclusion used a regex needing a leading path
      separator, which silently stopped working when the script was invoked with a relative path
      (e.g. `.`) instead of an absolute one; now checks path parts directly, verified to give the
      same count (109 orphans, 15 image-source false positives, out of 1086 `.tex` files) whether
      invoked with `.`, an absolute path, or no argument.

## Needs the user's go-ahead before running (no LaTeX compile without asking, per standing rule)

- [ ] `LaTeX/TODO.md` item #11: re-run the smoke-test compiles for `Main_Workshop_LLM_Presentation.tex`,
      `Main_Workshop_NLP_Presentation.tex`, and `Main_Workshop_NLP_Deep_Presentation.tex` with a
      much longer timeout, to confirm whether the growing `Overfull \hbox` pattern in the NLP
      workshop's log is a real non-converging bug or the known pre-existing scale issue.
- [ ] `LaTeX/TODO.md` item #12: retry the full `Main_Course_GenerativeAI_Presentation.tex` compile
      to test whether removing the 15 duplicate `\input` lines fixed the original runaway bug.
- [ ] `LaTeX/TODO.md` item #13: record the outcome of #11/#12 in `CLAUDE.md`'s Known Issues
      section, replacing the current bisection writeup.

## Sub-project trackers

- [`LaTeX/TODO.md`](LaTeX/TODO.md): GenerativeAI course restructuring (items #11-13 above are
  pulled from here; see that file for the full history of items #1-10, already done).
