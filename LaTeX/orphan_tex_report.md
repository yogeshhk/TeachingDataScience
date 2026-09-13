# Orphan `.tex` File Report

Generated 2026-09-13 by `find_orphan_tex.py` (same directory). "Orphan" means: exists on disk
under `LaTeX/`, but is not reachable via `\input{}`/`\include{}` from any `Main_*.tex` driver's
chain, transitively. Re-run the script any time to refresh these numbers; they will drift as
content is added, moved, or wired in.

**Snapshot: 109 orphans out of 1086 `.tex` files (10.0%)**, plus 15 files under `images/` that
the script excludes as false positives (see below).

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

## Bucket B: possible `_short` sibling drift, 17 files, needs per-pair checking, not a bulk move

The repo has a real `X.tex`/`X_short.tex` sibling convention (see `CLAUDE.md`'s "Sibling-file
sync rule") for genuine shorter-duration variants. Each of these ends in `_short` but its
un-suffixed sibling either doesn't exist, or exists and no longer references the short
version, so it can't be told apart from Bucket A by filename alone:

```
career_ai_roles_short.tex     ml_agri_short.tex
dl_intro_short.tex            ml_concepts_short.tex
dl_python_short.tex           ml_refs_short.tex
dnlp_intro_short.tex          nlp_embedding_short.tex
genai_intro_short.tex         nlp_refs_short.tex
llm_fromzero_short.tex        python_syntax_short.tex
rl_concepts_short.tex         rl_qlearning_short.tex
rl_conclusion_short.tex
rl_deepqlearning_short.tex
rl_intro_short.tex
```

**Recommendation:** for each, check whether a `_Short` driver already exists and just needs this
file wired back in, or whether it's genuinely abandoned (in which case it joins Bucket A).
Not done in this pass; deliberately deferred rather than guessed at 17 times.

---

## Bucket C: real, substantive content with no driver at all, 58 files

Grouped by theme. These are not junk: someone wrote real slides for these topics, but a visitor
following the README/COURSES.md navigation cannot currently reach any of them.

**AI tools, 10 files.** A near-complete standalone curriculum on AI coding tools, currently
invisible next to the existing Claude Code / OpenCode seminars in `COURSES.md`:
`ai_tools_claudecode_bestprac.tex`, `ai_tools_claudecode_demo_advanced.tex`,
`ai_tools_claudecode_demo_basic.tex`, `ai_tools_claudecode_intro.tex`,
`ai_tools_claudecode_setup.tex`, `ai_tools_claudecowork.tex`, `ai_tools_notebooklm.tex`,
`ai_tools_openwork.tex`, `ai_tools_sarvam_demo.tex`, `career_ai_tools.tex`

**Rasa chatbot example bots, 4 files.** The Rasa workshop's own catalog entry mentions "a full
IPL-bot walkthrough"; these look like alternate/earlier bot examples not currently wired into
that chain: `chatbot_rasa_chatbot_nlu_gstbot.tex`, `chatbot_rasa_chatbot_nlu_restaurant.tex`,
`chatbot_rasa_chatbot_slots_bookingbot.tex`, `chatbot_rasa_installdemo.tex`

**Applied AI in sensitive domains, 4 files:** `ai_healthcare.tex`, `chatbot_healthcare.tex`,
`dnlp_phishing.tex`, `dnlp_security.tex`

**Graph RAG implementation, 5 files.** `COURSES.md` lists a Graph RAG seminar but these
implementation-specific files aren't in its chain: `graph_rag_concepts.tex`,
`graph_rag_impl_fastrag.tex`, `graph_rag_impl_langchain.tex`, `graph_rag_impl_llamaindex.tex`,
`graph_rag_impl_neo4j.tex`

**Knowledge graphs, 2 files:** `kg_llm.tex`, `kg_overview.tex`

**Conference talk fragments, 2 files:** `langchain_devcon2025_start.tex`,
`langchain_devcon2025_end.tex`

**LLM agents, alternate frameworks, 4 files:** `llm_agents_impl_crewai.tex`,
`llm_agents_impl_google.tex`, `llm_agents_impl_smolagents.tex`, `llm_agents_mads.tex`

**LLM fine-tuning, alternate providers, 2 files:** `llm_finetuning_huggingface.tex`,
`llm_finetuning_openai.tex`

**Misc LLM, 4 files:** `llm_transformers_openai.tex`, `llm_promptengg_applications_marketing.tex`,
`llm_promptengg_thermal.tex`, `llm_concepts.tex`

**Deep learning, Keras variants, 3 files:** `dl_classification_cnn_keras.tex`,
`dl_intro_keras.tex`, `dl_rnn_keras.tex`

**Data/analytics, 2 files:** `data_evaluation.tex`, `data_visualization_churn.tex`

**Maths, 1 file:** `maths_calculus_integration.tex`

**ML for agriculture, applied bundle, 2 files:** `ml_agri_assignments.tex`, `ml_agri_refs.tex`
(`ml_agri_short.tex` is in Bucket B)

**MLCoEP-style assignments/demos, 3 files:** `ml_course_assign3_decisiontree_synthetic.tex`,
`ml_course_assign5_randomforest_creditscore.tex`, `ml_course_demo3_decisiontree_synthetic.tex`

**ML misc, 2 files:** `ml_mech_refs.tex`, `ml_tensorflow.tex`

**Course/venue meta fragments, 6 files:** `coep_course_logistics.tex`, `ds_course_intro.tex`,
`mlme_title.tex`, `about_me_seqseg.tex`, `ai_educators_technical_optional.tex`, `tedx.tex`

**Structural gap, 1 file.** `seminar_artificialintelligence_tools_content.tex` is a
`seminar_*_content.tex` aggregator file with no `Main_Seminar_*` driver pointing at it at all,
unlike every other seminar in the repo. Given the "AI tools" cluster above, this may be the
missing driver target for that whole cluster.

**Unused template, 1 file:** `template_homework.tex`

**Recommendation:** no bulk action here, this is a decision list, not a task list. Options per
item are: wire it into an existing seminar/workshop `\input` chain, give it (and its cluster)
its own new `Main_Seminar_*` driver, or move it to `_retired/` if it's genuinely abandoned. The
AI tools cluster plus `seminar_artificialintelligence_tools_content.tex` looks like the
highest-value single fix, since it's a nearly-complete mini-course sitting one driver file away
from being catalog-visible.
