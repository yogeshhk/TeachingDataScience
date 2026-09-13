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

- [x] Resolve Bucket B (17 possible `_short` sibling-drift files from `orphan_tex_report.md`).
      Done 2026-09-13: checked each against its sibling and any referencing content file.
      14 resolved and moved: 6 belonged entirely to an already-`_retired` seminar (5 RL `_short`
      files used only by `_retired/ml_reinforcementlearning_ad_hoc_seminar/...`, plus
      `llm_fromzero_short.tex` used only by `_retired/ai_chatgpt/...`) and were moved into those
      same `_retired/` subfolders to keep each family self-contained; 8 had zero references
      anywhere, live or commented (`dl_intro_short`, `dnlp_intro_short`, `genai_intro_short`,
      `ml_refs_short`, `nlp_embedding_short`, `nlp_refs_short`, `python_syntax_short`,
      `ml_concepts_short`), and were moved into `LaTeX/_retired/superseded_drafts/` alongside
      Bucket A. Verified after: orphan count dropped 75 -> 61, 0 of 279 drivers blocked.
      **3 left unresolved, flagged for your call** (all are commented-out `\input` lines sitting
      inside currently-live content files, so moving them is a content decision, not cleanup):
      - `career_ai_roles_short.tex`: commented out of the live `Main_Seminar_AI_Career_Short_*`
        deck while the full `career_ai_roles.tex` is live in the Full version. Looks like a
        deliberate "keep the short version shorter" choice, not drift.
      - `dl_python_short.tex`, `ml_agri_short.tex`: each commented out of a currently-live course
        content file (`seminar_deeplearning_content.tex`, `seminar_machinelearning_content.tex`)
        with an inline note describing real intended content, but neither has a non-`_short`
        sibling on disk at all, so it's unclear whether these are unfinished drafts worth
        finishing or abandoned ideas worth retiring.
- [x] Resolve Bucket C (58 real-content-gap files from `orphan_tex_report.md`).
      Done 2026-09-13: a follow-up grep (checking every filename inside comments too, not just
      live `\input`s) found the original "no driver at all" framing was mostly wrong. Reclassified:
      **46 of 58 need no action at all** (already `% \input{...}`'d, with a descriptive comment,
      inside a currently-live content file, e.g. the whole Graph RAG and Rasa-bot clusters: this
      is just the repo's normal backlog of drafted-but-unactivated slides, not an orphan problem).
      **7 were genuinely dead and got moved**: `kg_llm.tex`/`kg_overview.tex` (superseded by the
      live, split-up `kg_llm_intro/_conclusions/_refs`), `ml_tensorflow.tex`, and
      `template_homework.tex` (zero references anywhere) went to
      `LaTeX/_retired/superseded_drafts/`; `about_me_seqseg.tex`, `tedx.tex`, and
      `ai_tools_sarvam_demo.tex` (each still actively `\input` by content already living under
      `_retired/`) went into that same `_retired/` subfolder as their consumer.
      **5 left flagged, needs the maintainer's call**: `seminar_artificialintelligence_tools_content.tex`
      turned out not to be "just missing a driver": every line in it is commented out, two of its
      references (`ai_tools_opencode_intro`/`_demo`) would duplicate content already live
      elsewhere (the exact pattern already documented and fixed once in `LaTeX/TODO.md` for the
      GenerativeAI course), and one reference (`ai_tools_claudecode`) points to a file that
      doesn't exist. Its 3 unique files (`ai_tools_claudecowork.tex`, `ai_tools_notebooklm.tex`,
      `ai_tools_openwork.tex`) plus `career_ai_tools.tex` are referenced only by this draft.
      Options are: finish it as a real seminar (dropping the duplicate/stale lines first), fold
      the unique files elsewhere, or retire the cluster; not resolved here.
      Verified after all moves: orphan count 61 -> 54, `latex-audit -Mode inputs` still 0 of 279
      drivers blocked.

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
