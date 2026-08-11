# TODO.md: Repo-Level Index

Index only, not a duplicate: points at where the real plans/status live so "what's next" from the
repo root gives a global picture without digging through every subfolder.

## Actively developed

| Area | TODO / status | Notes |
|---|---|---|
| `Code/mlcoep/` (COEP AIML course) | See `CLAUDE.md` "19-session restructuring" note | Course in progress (Jul 16 - Nov 30, 2026), updated session-by-session, 2 days/week. Setup (Python/conda) done; datasets/notebooks/assignments/exams dropped from scope (Aug 2026) in favor of slides + in-class code walkthroughs, so `Code/mlcoep/TODO_MLCOEP_TECHNICAL.md` was removed. Slide content: 19 independent session drivers, `Main_Seminar_MLCoEP_Session_<N>_<ShortName>_{Presentation,CheatSheet}.tex` (compiled via `make_all_sessions.bat`/`make_all_cheatsheets.bat`; the old `/prep-mlcoep-session` skill is retired). No running todo.md; per-session `/upgrade-deck` passes continue individually as bandwidth allows (Sessions 5-6 done, see `CLAUDE.md`). **Recompile backlog: CLEARED (verified on disk 2026-08-07).** The "Quick Check quiz format changed" edit (see `CLAUDE.md`) touched 15 topic files and this entry used to list Sessions 2,3,4,10,12,13,14,15,18 as still needing a rebuild. All 19 MLCoEP session PDFs (Presentation and CheatSheet, 38 files) are now newer than every one of those 15 topic files, so none of them is stale: Sessions 1,11,14,15,16,17,18 were rebuilt 2026-08-06, Sessions 2,4,5,7,9,10,12,13 on 2026-08-07 morning, and Sessions 3,6,8,19 on 2026-08-07 afternoon during the 3-column CheatSheet pass. The "12 non-MLCoEP decks sharing the same files" clause is also moot: those drivers (`Main_Seminar_ML_*`, `Main_Seminar_Python_Overview_*`, `Main_Seminar_AI-ML_*`, `Main_Seminar_Python_Advanced_DataLibs_*`) have no PDFs checked in at all, so there is nothing on disk to be stale; they get built on demand. Only the 38 MLCoEP PDFs are kept here. |

## Completed/maintained courses (no dedicated TODO.md tracked here)

The Machine Learning / Deep Learning / Generative AI / Maths for ML / Python courses under `LaTeX/`
have had recent restructuring and maintenance work (June-Aug 2026); their own `CLAUDE.md` documents
these updates. They have no dedicated running TODO.md (by this repo's own convention, working todos
get deleted once each pass completes). The wider `Code/` project catalog (LangChain, LangGraph, RAG,
GNN, fine-tuning, etc.) is maintained but not actively scheduled. Add an entry above (and a TODO.md
in the relevant subfolder) if either track becomes active-project-scheduled work again.
