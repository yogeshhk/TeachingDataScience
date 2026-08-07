# TODO.md: Repo-Level Index

Index only, not a duplicate: points at where the real plans/status live so "what's next" from the
repo root gives a global picture without digging through every subfolder.

## Actively developed

| Area | TODO / status | Notes |
|---|---|---|
| `Code/mlcoep/` (COEP AIML course) | See `CLAUDE.md` "19-session restructuring" note | Course in progress (Jul 16 - Nov 30, 2026), updated session-by-session, 2 days/week. Setup (Python/conda) done; datasets/notebooks/assignments/exams dropped from scope (Aug 2026) in favor of slides + in-class code walkthroughs, so `Code/mlcoep/TODO_MLCOEP_TECHNICAL.md` was removed. Slide content: 19 independent session drivers, `Main_Seminar_MLCoEP_Session_<N>_<ShortName>_{Presentation,CheatSheet}.tex` (compiled via `make_all_sessions.bat`/`make_all_cheatsheets.bat`; the old `/prep-mlcoep-session` skill is retired). No running todo.md; per-session `/upgrade-deck` passes continue individually as bandwidth allows (Sessions 5-6 done, see `CLAUDE.md`). **Pending recompile (Aug 2026):** the "Quick Check quiz format changed" edit (see `CLAUDE.md`) touched 15 topic files consumed by Sessions 2,3,4,5,6,7,9,10,12,13,14,15,18 — Sessions 5,6,7,9 recompiled clean and verified, but Sessions 2,3,4,10,12,13,14,15,18 (and 12 non-MLCoEP decks sharing the same files) still need a recompile + visual check before their PDFs on disk can be called current. |

## Completed/maintained courses (no dedicated TODO.md tracked here)

The Machine Learning / Deep Learning / Generative AI / Maths for ML / Python courses under `LaTeX/`
have had recent restructuring and maintenance work (June-Aug 2026); their own `CLAUDE.md` documents
these updates. They have no dedicated running TODO.md (by this repo's own convention, working todos
get deleted once each pass completes). The wider `Code/` project catalog (LangChain, LangGraph, RAG,
GNN, fine-tuning, etc.) is maintained but not actively scheduled. Add an entry above (and a TODO.md
in the relevant subfolder) if either track becomes active-project-scheduled work again.
