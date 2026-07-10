# TODO: Audit short/full topic-file pairs for the comment-sync principle

Goal: across the whole `LaTeX/` directory, check whether each `X_short.tex` file
is a true subset-by-commenting of its full `X.tex` counterpart — i.e. every frame
in the full file also exists in the short file (active or commented, same
content) — versus files that have drifted apart or were authored independently.
No changes yet; this is analysis-only, to be done in the next session.

## Pairs with a full counterpart (24) — to be checked

For each, verify: does every frame title/content in `X.tex` also appear somewhere
in `X_short.tex` (live or commented)? Flag any frame present in one but not the
other.

- [ ] `chatbot_intro.tex` / `chatbot_intro_short.tex`
- [ ] `data_intro.tex` / `data_intro_short.tex`
- [ ] `data_refs.tex` / `data_refs_short.tex`
- [ ] `dl_intro.tex` / `dl_intro_short.tex` — known from earlier in this session:
      full has 57 active frames, short has 67 total (41 active + 26 commented) —
      short appears to have *more* content than full, worth checking closely
- [ ] `dl_refs.tex` / `dl_refs_short.tex`
- [ ] `dnlp_intro.tex` / `dnlp_intro_short.tex`
- [ ] `ml_refs.tex` / `ml_refs_short.tex`
- [ ] `nlp_embedding.tex` / `nlp_embedding_short.tex`
- [ ] `nlp_refs.tex` / `nlp_refs_short.tex`
- [ ] `python_refs.tex` / `python_refs_short.tex`
- [ ] `python_syntax.tex` / `python_syntax_short.tex`
- [ ] `llm_fromzero.tex` / `llm_fromzero_short.tex`
- [ ] `chatgpt_intro.tex` / `chatgpt_intro_short.tex`
- [ ] `rl_concepts.tex` / `rl_concepts_short.tex`
- [ ] `rl_conclusion.tex` / `rl_conclusion_short.tex`
- [ ] `rl_deepqlearning.tex` / `rl_deepqlearning_short.tex`
- [ ] `rl_qlearning.tex` / `rl_qlearning_short.tex`
- [ ] `genai_intro.tex` / `genai_intro_short.tex`
- [ ] `rl_intro.tex` / `rl_intro_short.tex`
- [ ] `ml_intro.tex` / `ml_intro_short.tex` — known from earlier in this session:
      full has 1745 lines, short has 1757 — likely close but worth confirming
- [ ] `ai_intro_tech.tex` / `ai_intro_tech_short.tex` — created this session
      (2026-Oct), known good by construction
- [ ] `career_ai_prep.tex` / `career_ai_prep_short.tex` — created this session,
      known good by construction
- [ ] `career_ai_roles.tex` / `career_ai_roles_short.tex` — created this session,
      known good by construction
- [ ] `career_ai_personas.tex` / `career_ai_personas_short.tex` — created this
      session, known good by construction

## `_short.tex` files with NO full counterpart (7) — different category

These either predate the comment-sync convention, were authored standalone, or
their full sibling was renamed/removed at some point. Not part of the sync check
above — just flag for awareness, decide case-by-case whether a full counterpart
should exist or whether these are intentionally standalone:
- [ ] `ml_mech_short.tex` (no `ml_mech.tex`)
- [ ] `python_ai_short.tex` (no `python_ai.tex`)
- [ ] `ml_agri_short.tex` (no `ml_agri.tex`)
- [ ] `python_kids_short.tex` (no `python_kids.tex`)
- [ ] `python_engineering_short.tex` (no `python_engineering.tex`)
- [ ] `dl_python_short.tex` (no `dl_python.tex`)
- [ ] `python_intro_short.tex` (no `python_intro.tex`)

## Proposed verification method (next session)

To avoid reading all 24 pairs' full content (expensive), use a two-pass approach:

1. **Fast pass (scriptable)**: for each pair, extract the set of `\frametitle{...}`
   values from both files (matching both `\begin{frame}...` and `% \begin{frame}...`
   commented lines via a relaxed regex), and diff the two title sets.
   - Titles in `X.tex` missing from `X_short.tex`'s title set (live or commented)
     = a real gap, the short file doesn't have that content at all.
   - Titles only in `X_short.tex` and not in `X.tex` = short has content the full
     doesn't (either the full lost a frame the short still carries, or the short
     was extended independently).
   - Matches on title alone don't guarantee identical body content — title
     collisions with divergent bodies would need a closer look.
2. **Deep pass (manual read)**: only for pairs the fast pass flags as mismatched,
   read both files fully and compare frame-by-frame content, not just titles.

Report back: a per-pair verdict (in sync / drifted / needs a closer look) plus a
short list of the specific missing or extra frames for anything flagged.
