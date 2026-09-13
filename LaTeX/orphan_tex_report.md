# Orphan `.tex` File Report

Generated 2026-09-13 by `find_orphan_tex.py` (same directory). "Orphan" means: exists on disk
under `LaTeX/`, but is not reachable via `\input{}`/`\include{}` from any `Main_*.tex` driver's
chain, transitively. Re-run the script any time to refresh these numbers; they will drift as
content is added, moved, or wired in.

**Original snapshot: 109 orphans out of 1086 `.tex` files (10.0%)**, plus 15 files under
`images/` excluded as false positives (see below). **Updated 2026-09-13, fully worked through:**
Bucket A (34 files), 14 of Bucket B's 17 files, and 7 of Bucket C's 58 files are now moved into
`_retired/` (see `TODO.md` for exactly what moved where). Current count: **54 orphans
remaining**, and none of them need further mechanical cleanup: 3 are flagged Bucket B items and
5 are a flagged Bucket C cluster (both need a maintainer content decision, not cleanup), and the
other 46 are Bucket C items that turned out to already be correctly homed (commented out,
pending activation, in a live content file) rather than actually orphaned.

---

## Not actually orphans (excluded by the script)

`images/*.tex` and `images/tikz/*.tex` (15 files) are standalone TikZ/PGF sources, compiled
independently to PDF and then pulled in via `\includegraphics` elsewhere (`LaTeX/images/*.pdf`
are source assets, not build output, per `CLAUDE.md`). They will never appear in a `\input`
chain by design. No action needed on these.

---

## Bucket A: superseded drafts, 34 files, candidates for `LaTeX/_retired/`

Filenames self-flag as dead via `_old`, `_v0`/`_v1`/`_v2`, `_del`, `_depricated`, `_tooold`,
`_bk`, or a dated one-off (`_2024`). This is the list that TODO item 5 ("archive superseded
drafts") will act on:

```
ai_intro_2024.tex                      llm_intro_v0.tex
ai_intro_old.tex                       llm_intro_v1.tex
ai_tools_opencode_old.tex              llm_intro_v2.tex
dl_cnn_keras_depricated.tex            llm_promptengg_conclusions_v0.tex
langchain_intro_short_del.tex          llm_promptengg_intro_v0.tex
langchain_onepager_old.tex             llm_promptengg_overview_v0.tex
langchain_overview_del.tex             llm_promptengg_sandwich_v0.tex
langgraph_intro_old.tex                llm_promptengg_techniques_v0.tex
llm_agents_conclusions_v1.tex          llm_rag_advanced_v0.tex
llm_agents_conclusions_v2.tex          llm_rag_conclusions_v0.tex
llm_agents_impl_autogen_old.tex        llm_rag_framework_v0.tex
llm_agents_impl_v1.tex                 llm_rag_intro_v0.tex
llm_agents_intro_v0_full.tex           llm_rag_multimodal_v0.tex
llm_agents_intro_v0_small.tex          llm_transformers_conclusion_old.tex
llm_agents_intro_v1.tex                llm_transformers_howitworks_old.tex
llm_agents_intro_v2.tex                llm_transformers_howitworks_tooold.tex
llm_evaluation_intro_bk.tex            llm_transformers_intro_old.tex
```

**Recommendation:** move to `LaTeX/_retired/`, don't delete (matches the repo's existing
convention for dead/archived content). Worth a quick skim before moving in bulk in case any one
of these is actually still `\input` from somewhere my script doesn't check (it only follows
`Main_*.tex` chains, not arbitrary cross-references).

---

## Bucket B: possible `_short` sibling drift, 17 files: RESOLVED except 3 flagged items

Checked each against its sibling and any file referencing it (live or commented).

**14 moved to `_retired/`:**
- `rl_concepts_short.tex`, `rl_conclusion_short.tex`, `rl_deepqlearning_short.tex`,
  `rl_intro_short.tex`, `rl_qlearning_short.tex`: used only by the already-`_retired`
  `ml_reinforcementlearning_ad_hoc_seminar/seminar_reinforcementlearning_content.tex`. Moved
  into that same `_retired/` subfolder to keep the family self-contained.
- `llm_fromzero_short.tex`: used only by three content files already living under
  `_retired/ai_chatgpt/`. Moved there.
- `dl_intro_short.tex`, `dnlp_intro_short.tex`, `genai_intro_short.tex`, `ml_refs_short.tex`,
  `nlp_embedding_short.tex`, `nlp_refs_short.tex`, `python_syntax_short.tex`,
  `ml_concepts_short.tex`: zero references anywhere, live or commented (their un-suffixed
  siblings are live via a different, unrelated content file in every case). Moved into
  `LaTeX/_retired/superseded_drafts/` alongside Bucket A.

**3 left in place, flagged for a maintainer decision** (each is a commented-out `\input` inside a
currently-*live* content file, so this is a content call, not cleanup):
- `career_ai_roles_short.tex`: commented out of the live `Main_Seminar_AI_Career_Short_*` deck
  while the full `career_ai_roles.tex` is live in the Full version. Reads as a deliberate "keep
  the short version shorter" choice already in effect, not drift.
- `dl_python_short.tex` (commented in `seminar_deeplearning_content.tex`, "Python for DL
  (short)"), `ml_agri_short.tex` (commented in `seminar_machinelearning_content.tex`, "ML for
  agriculture"): neither has a non-`_short` sibling on disk at all, so unclear whether these are
  unfinished drafts worth finishing or abandoned ideas worth retiring.

---

## Bucket C: content with no driver, 58 files originally: mostly not actually a gap

The original pass judged these purely by driver-chain reachability, which made them look like
unclaimed content. A follow-up grep for each filename (including inside comments) across every
`seminar_*_content.tex`/`workshop_*_content.tex`/`course_*_content.tex` file told a different,
much more mundane story for most of them. Reclassified into four groups:

### C1: already homed, just commented out pending activation (46 files): no action needed

Every one of these already has an explicit `% \input{...}` line, usually with a descriptive
comment, inside a **currently-live** content file. This is the repo's normal working state for
drafted-but-not-yet-enabled material, not an orphan problem:

- `ai_healthcare.tex` (in `seminar_ai_for_all_tech_content.tex`)
- `chatbot_healthcare.tex`, `chatbot_rasa_chatbot_nlu_gstbot.tex`,
  `chatbot_rasa_chatbot_nlu_restaurant.tex`, `chatbot_rasa_chatbot_slots_bookingbot.tex`,
  `chatbot_rasa_installdemo.tex` (in `seminar_chatbot_content.tex` and/or
  `workshop_chatbot_rasa_content.tex`)
- `dl_intro_keras.tex`, `dl_classification_cnn_keras.tex`, `dl_rnn_keras.tex` (in
  `seminar_deeplearning_content.tex`)
- `data_evaluation.tex`, `data_visualization_churn.tex`, `ml_mech_refs.tex`,
  `ml_agri_assignments.tex`, `ml_agri_refs.tex`, `mlme_title.tex`, `ds_course_intro.tex`,
  `coep_course_logistics.tex` (all in `seminar_machinelearning_content.tex`)
- `llm_finetuning_openai.tex`, `llm_finetuning_huggingface.tex` (in
  `seminar_llm_finetuning_content.tex`)
- `llm_transformers_openai.tex` (in `seminar_llm_transformers_content.tex`)
- `maths_calculus_integration.tex` (in
  `seminar_maths4ml_calculus_derivatives_optimization_content.tex`, marked "TBD")
- `graph_rag_concepts.tex`, `graph_rag_impl_fastrag.tex`, `graph_rag_impl_neo4j.tex`,
  `graph_rag_impl_langchain.tex`, `graph_rag_impl_llamaindex.tex` (in
  `seminar_graph_rag_content.tex`)
- `llm_promptengg_applications_marketing.tex`, `llm_promptengg_thermal.tex` (in
  `seminar_llm_promptengg_content.tex` / `seminar_llm_genai_content.tex`)
- `dnlp_phishing.tex`, `dnlp_security.tex` (in `workshop_deepnlp_content.tex`)
- `ml_course_demo3_decisiontree_synthetic.tex`, `ml_course_assign3_decisiontree_synthetic.tex`,
  `ml_course_assign5_randomforest_creditscore.tex` (in `course_machinelearning_content.tex`)
- `ai_educators_technical_optional.tex` (in `seminar_ai_for_educators_content.tex`)
- `llm_agents_mads.tex`, `llm_agents_impl_crewai.tex`, `llm_agents_impl_smolagents.tex`,
  `llm_agents_impl_google.tex` (in `seminar_llm_agents_content.tex` /
  `workshop_llm_agents_content.tex`)
- `langchain_devcon2025_start.tex`, `langchain_devcon2025_end.tex` (in
  `seminar_llm_langchain_content.tex` / `workshop_llm_langchain_content.tex`)
- `llm_concepts.tex` (in the live `seminar_llm_intro_content.tex`)
- `ai_tools_claudecode_intro.tex`, `ai_tools_claudecode_setup.tex`,
  `ai_tools_claudecode_demo_basic.tex`, `ai_tools_claudecode_bestprac.tex`,
  `ai_tools_claudecode_demo_advanced.tex` (in `seminar_ai_claudecode_content.tex`, itself live
  via `Main_Seminar_AI_HandsOn_ClaudeCode_Presentation.tex`/`CheatSheet.tex`: initially miscounted
  as part of the "AI tools" cluster below before checking where they're actually referenced)

**No action taken.** Activating any of these is a content decision (is the draft finished, does
it read well) for whoever owns that seminar, not a structural fix.

### C2: genuinely dead, zero reference anywhere, now moved (4 files)

`kg_llm.tex` and `kg_overview.tex` (superseded by the live, more granular `kg_llm_intro.tex`/
`kg_llm_conclusions.tex`/`kg_llm_refs.tex` in `workshop_graph_kg_content.tex`), `ml_tensorflow.tex`,
and `template_homework.tex` had no reference anywhere, live or commented. Moved into
`LaTeX/_retired/superseded_drafts/` alongside Buckets A and B's zero-reference files.

### C3: retired-family leftovers, now moved (3 files)

`about_me_seqseg.tex`, `tedx.tex`, and `ai_tools_sarvam_demo.tex` were each still actively (not
commented) `\input` by content that already lives under `_retired/` (`_retired/llm_seqseg/`,
`_retired/ai_chatgpt/`, `_retired/ai_sarvam/` respectively). Moved into those same folders to
keep each family self-contained, matching the Bucket A/B precedent.

### C4: genuinely inconsistent, flagged for a real decision (5 files)

`seminar_artificialintelligence_tools_content.tex` is a `seminar_*_content.tex` aggregator with
no driver at all, unlike every other seminar in the repo, and it is not simply "missing a
driver": every line in it is commented out, and two of its eight references
(`ai_tools_opencode_intro`, `ai_tools_opencode_demo`) point to files that are already live
elsewhere (`Main_Seminar_AI_HandsOn_OpenCode_Presentation.tex`), so activating them here would
duplicate content, echoing the exact duplicate-`\input` pattern `LaTeX/TODO.md` already
documents and fixed once for the GenerativeAI course. A third reference,
`\input{ai_tools_claudecode}` (no such file exists), is simply stale. The three files unique to
this draft, `ai_tools_claudecowork.tex`, `ai_tools_notebooklm.tex`, `ai_tools_openwork.tex`, plus
`career_ai_tools.tex`, are referenced only here.

**Left untouched, needs the maintainer's call:** finish this as a real standalone "AI tools
survey" seminar (dropping the two OpenCode duplicate lines and the stale ClaudeCode line first),
fold its three unique files into other existing seminars instead, or retire the whole cluster.
Not something to guess at.
