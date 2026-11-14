# Project: demo

> Status: built: yes | measured: shows the measured score from `quantum-wiki` | reviewed-by-author: yes (the five questions were picked by the author) | needs LM Studio: yes

The offline live demo page for the Frugal AI talk (Dec 12, 2026). One Streamlit screen: five question buttons, then
"Base model" next to "Base model + RAG" (structure-aware retrieval) for the same question, and one line with the
measured score. It runs on a laptop with no internet; the page refuses every connection that is not to this machine,
so switching the WiFi off is the proof.

## Problem and scope
- In: one of five fixed questions about the AICTE Minor in Quantum Technologies syllabus.
- Out: two answers side by side, so an audience sees what retrieval adds to a small model.
- Not in scope: the earlier Marathi and router sections (kept in `_backup/`), `workflow-agent` and `voice-form`.

## Where it lives
This folder is in the TeachingDataScience repo because the Frugal AI deck is here. The projects it shows
(`quantum-wiki`, `workflow-agent`, `finetune-slm`, `common`) live in the FrugalAI repo. `demo_lib.py` finds that
repo next to this one (`D:\Yogesh\GitHub\FrugalAI`), or from the `FRUGALAI_REPO` environment variable if the clone is
elsewhere.

## Run it
1. Start LM Studio and its server. Load Qwen3 1.7B and nomic-embed-text; unload other models. Close browsers first (RAM).
2. From this folder: `conda run -n genai streamlit run app.py`.
3. Tests, offline: `python -m pytest tests -q`.

Answers vary a little from run to run.

## What is in here
| Path | Purpose |
|---|---|
| `app.py` | The talk page |
| `demo_lib.py` | Helpers: socket guard, model checks, retrieval calls |
| `capture.py` | Takes the five screenshots of the page, `python capture.py <out_dir>` with the page running |
| `app_faq.py` | Second page: is a concept already in the author's FAQs. Parked, needs the private FAQ folder |
| `make_graph_figures.py` | Redraws the workflow and path pictures. Parked |
| `ANALYSIS.md` | Why the demo looks the way it does, and the trial run |
| `_backup/` | The earlier three-section page and its capture script |
| `tests/` | Offline tests |

## Evaluation and limits
The score line comes from `code/quantum-wiki/` in the FrugalAI repo (one run, one machine). The question set was drafted by Claude and
reviewed by the author on 2026-10-31. See `ANALYSIS.md` for what the demo does not show.

## Open actions
See `TODO.md`.
