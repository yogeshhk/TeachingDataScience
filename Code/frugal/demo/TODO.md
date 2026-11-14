# TODO (demo): offline live demo for the Frugal AI talk (Dec 12, 2026)

Run: LM Studio serving Qwen3 1.7B and nomic-embed (start its server; unload other models), then `conda run -n genai streamlit run app.py` from this folder. Close browsers first (RAM). Tests: `python -m pytest tests -q` here.
The page is one screen: five question buttons, then "Base model" against "Base model + RAG" (structure-aware retrieval) and one line with the measured score. The answers vary a little from run to run. Analysis and the trial run: `ANALYSIS.md`. The earlier three-section page (Marathi, router) and its capture script are in `_backup/`.
Screenshots: `python capture.py <out_dir>` with the page running (bare view, five PNGs `frugal_rag_q1` to `q5`).
Decided: Marathi fine-tune and router are out of the demo; the five questions were picked by the author after a trial (ANALYSIS.md section 8); `workflow-agent` and `voice-form` left out; AIOOLL not used.

## Open
None. The demo is ready for the talk: question sets signed off by the author and scores re-checked.

## Parked (not part of the Dec 12 talk)
- Second page `app_faq.py` (is a concept already in my FAQs, and where else does it appear): needs LM Studio serving nomic-embed only; run `conda run -n genai streamlit run app_faq.py`. It reads the real QCNP folder; the author agreed that real FAQ entry names appear in the five `frugal_faq_q*` pictures in the deck. `make_graph_figures.py` redraws the workflow and path pictures (needs LM Studio with nomic-embed and Qwen3 1.7B; `--reuse` redraws from the saved run).
- `capture.py` and the page use a socket guard copied from FrugalAI's `code/voice-form/voiceform/netguard.py`; merge the two into `code/common/frugalai_common/` in the FrugalAI repo once voice-form is picked up again.
- Marathi fine-tune: rerun with a better setup later (see `ANALYSIS.md`); voice-form: see `code/voice-form/TODO.md` in the FrugalAI repo.
