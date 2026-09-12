# LaTeX/ Action Items: GenerativeAI Course Restructuring

Investigated 2026-09-12 after `Main_Course_GenerativeAI_{Presentation,CheatSheet}` failed to
compile (see `CLAUDE.md`'s "Known issues" for the full bisection writeup). Root cause: of this
course's 51 `\input` lines, only 3 are workshop-level (every other course in this repo either has
zero course-level extras, like Maths4ML/Python, or a modest handful, like ML/DeepLearning); the
other 48 bypass the Workshop/Seminar structure entirely, and at least 9 of those are **literal
duplicates** of files already pulled in via `workshop_llm_content`'s own seminars. Work through
these one at a time, not all at once -- confirm each item before moving to the next.

## 1. Verify no further hidden duplicates -- DONE (2026-09-12)

- [x] Checked all 43 active course-level `\input` targets against every seminar/workshop file
      reachable from `course_generativeai_content.tex`'s actual chain (not just
      `workshop_llm_content`, also `workshop_naturallanguageprocessing_content` and
      `workshop_deepnlp_content`). Found **5 more confirmed duplicates** beyond the original 10:
      `llm_evaluation_refs` (already via `seminar_llm_evaluation_content`), `llm_finetuning_refs`
      (via `seminar_llm_finetuning_content`), `llm_rag_refs` (via `seminar_llm_rag_content`),
      `nlp_refs` (via `seminar_nlp_advanced_content`), `dnlp_refs` (directly in
      `workshop_deepnlp_content.tex` itself). Total confirmed duplicates: **15**, folded into
      item 2 below.
- [x] Every other filename that also appears elsewhere in the repo traced back to a genuinely
      separate, independently-compiled standalone deck (`Main_Workshop_LLM_LangGraph_*`,
      `_LLM_LangChain_*`, `_LLM_Agents_*`, `Main_Seminar_LLM_GenAI_*`,
      `Main_RefCard_PromptEngineeing`, etc.), not reachable from this course's chain -- confirmed
      genuinely unique, not duplicates. This validates the file's own "(not in workshop_llm)"
      comments for LangGraph and Advanced RAG.

## 2. Remove the 15 confirmed duplicate `\input` lines -- DONE (2026-09-12)

- [x] Removed from `course_generativeai_content.tex`, with comments explaining each removal and
      pointing back to this item:
      - `chatgpt_intro` (already directly in `workshop_llm_content.tex`)
      - `langchain_intro`, `langchain_framework`, `langchain_conclusions`, `langchain_refs`
        (already via `seminar_llm_langchain_content`)
      - `llamaindex_intro`, `llamaindex_impl` (already via `seminar_llm_llamaindex_content`)
      - `llm_agents_intro`, `llm_agents_conclusions`, `llm_agents_refs` (already via
        `seminar_llm_agents_content`)
      - `llm_evaluation_refs` (already via `seminar_llm_evaluation_content`)
      - `llm_finetuning_refs` (already via `seminar_llm_finetuning_content`)
      - `llm_rag_refs` (already via `seminar_llm_rag_content`)
      - `nlp_refs` (already via `seminar_nlp_advanced_content`)
      - `dnlp_refs` (already directly in `workshop_deepnlp_content.tex`)
      All reach the course already via the proper Workshop/Seminar path; only the flat,
      bypass-everything course-level copy needs removing.

## 3. Home the 4 NLP extras into the existing NLP workshop's seminars -- DONE (2026-09-12)

- [x] `nlp_libraries` -> `seminar_nlp_basics_content.tex` (Intro section). `nlp_clustering` ->
      `seminar_nlp_advanced_content.tex` (Topic Modeling section, alongside the other unsupervised
      technique). `nlp_classification_gensim_movie_sentiment` and `nlp_twittersentiment_nltk` ->
      `seminar_nlp_ml_content.tex` (Text Classification section). Removed from the course-level
      extras list (course file now only has `genai_conclusions`/`genai_refs` left, see item 2's
      final state below).

## 4. Home the 6 DeepNLP extras directly into `workshop_deepnlp_content.tex` -- DONE (2026-09-12)

- [x] `dnlp_intro_gensim`, `dnlp_gensim_embedding`, `dnlp_bert_embedding`,
      `dnlp_tensorflow_embedding` -> Embedding section (alongside `dnlp_wordvectors_adv`).
      `nlp_pytorch`, `dnlp_tensorflow_chatbot` -> Implementations section (alongside
      `dnlp_textgeneration_keras`). This workshop has no seminar layer, so they join its existing
      flat `\input` list directly. Removed from the course-level extras list.

## 5. Fold the prompt-engineering extras into the existing seminar -- DONE (2026-09-12)

- [x] `llm_promptengg_overview`, `llm_promptengg_readyref` -> Introduction section (alongside
      `llm_promptengg_intro`). `llm_promptengg_applications_education` -> Use-cases section
      (alongside `llm_promptengg_applications_generic`). Removed from the course-level extras list.

## 6. Fold the Advanced RAG extras into the existing seminar -- DONE (2026-09-12)

- [x] Added a new "Advanced" section to `seminar_llm_rag_content.tex` (between Implementation and
      Conclusions) holding `llm_rag_advanced` and `llm_rag_vertexai`. Removed from the course-level
      extras list.

## 7. Give LangGraph its own seminar inside `workshop_llm_content` -- DONE (2026-09-12)

- [x] Turns out `seminar_llm_langgraph_content.tex` **already existed** (also used by a separate
      standalone `workshop_llm_langgraph_content.tex` deck) -- no new file needed, just
      `\input{seminar_llm_langgraph_content}` added to `workshop_llm_content.tex`, right after the
      LlamaIndex seminar. Removed `langgraph_intro/_conclusions/_refs` from the course-level
      extras list.

## 8. Give the domain-specific ChatGPT applications a proper home -- DONE (2026-09-12)

- [x] Added `chatgpt_applications`, `chatgpt_applications_bdo/_imi/_hr/_journalism` into
      `seminar_llm_applications_content.tex`'s Applications section, alongside its existing
      `llm_applications`. Removed from the course-level extras list.

**Final state of `course_generativeai_content.tex`** after items 2-8: down from 51 `\input` lines
to just 5 -- the 3 workshops, plus `genai_conclusions`/`genai_refs` (the only genuinely course-wide
content, confirmed unique in item 9). Verified via `latex-audit -Mode inputs`: chain still fully
resolves, 0 blocked.

## 9. Leave course-level wrap-up files alone -- DONE (2026-09-12, verified by item 1)

- [x] `genai_conclusions` and `genai_refs` confirmed genuinely unique to the Course level (the
      other five files originally listed here -- `nlp_refs`, `dnlp_refs`, `llm_finetuning_refs`,
      `llm_rag_refs`, `llm_evaluation_refs` -- turned out to be duplicates instead, moved to item
      2). A course-wide summary/references section is legitimate Course-layer content; no action
      needed on these two.

## 10. Add the "don't casually compile the combined driver" warning -- DONE (2026-09-12)

- [x] Added to both `Main_Course_GenerativeAI_Presentation.tex` and `_CheatSheet.tex`, matching
      the precedent from `Main_Course_MLCoEP_*` (root `CLAUDE.md`) and QCNP's Full Workshop driver.

## 11. Smoke-test every touched workshop/seminar individually -- IN PROGRESS (started 2026-09-12)

- [ ] Recompiling `Main_Workshop_LLM_Presentation.tex`, `Main_Workshop_NLP_Presentation.tex`, and
      `Main_Workshop_NLP_Deep_Presentation.tex` (all 3 touched by items 3-8) in the background,
      300s timeout each. **Not yet confirmed finished/clean as of this checkpoint** -- check
      `/tmp/wf_llm.log`, `/tmp/wf_nlp.log`, `/tmp/wf_dnlp.log` (or their exit codes) next session
      if not already done, before trusting items 3-8's edits are compile-safe.

## 12. Retry the full Course compile once, to test the root-cause fix

- [ ] With the 9 duplicates removed, attempt `Main_Course_GenerativeAI_Presentation.tex` again
      (background, generous timeout) to see whether removing the duplicate `\input`s actually
      resolved the runaway `Overfull \hbox` loop. This is the empirical test of the whole
      hypothesis in `CLAUDE.md`'s "Known issues" section.

## 13. Update `CLAUDE.md`'s known-issues note with the outcome

- [ ] Record whichever result item 12 produces (fixed, or still broken with new findings) in
      place of the current bisection writeup.
