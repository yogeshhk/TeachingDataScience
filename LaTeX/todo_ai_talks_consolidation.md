# AI / ChatGPT / Career talks consolidation

## STATUS AS OF 2026-08-21 (paused here, resume by reading this block first)

**Done, all verified clean via repo-wide grep sweeps (zero dangling `\input`/link references):**
Phase 1-4 rename of the AI/ChatGPT/Career family (Cluster 1); Graph/LLM/NLP/ML seminar
folding-or-retiring audit, all 7 buckets (Cluster 2); Tech/Data small-cluster audit, Mentoring
retired (Cluster 3); GCP retired, AI_Overview_*/AI_ChatGPT_* unified into `AI_For_<X>` with
ChatGPT content folded in, Sarvam retired, ClaudeCode/OpenCode renamed to `AI_HandsOn_*`
(Cluster 4). `COURSES.md` and `CLAUDE.md` are in sync with every rename/retirement below;
`README.md` was checked and doesn't name any of these decks directly, so needed no changes.

**All 6 open items clarified with the user (2026-08-21, same day), then executed autonomously
(same day, user explicitly authorized "go ahead without asking me anything") — items 1, 2, 3, 4, 5
are DONE; item 6 deliberately held, see below.**

1. ~~Recover pre-`_retired/`-convention deletions~~ — **DECIDED: no.** Left Text Mining, both
   Knowledge Graph seminars, and Mentoring deleted. No action taken.
2. **AI_For_Educators content revival — DONE.** Uncommented the 5 dormant sections
   (Introduction/Techniques/Demo/Use-cases/Conclusions, all already-authored — `llm_promptengg_
   intro`, `_techniques_basic`, `_sandwich`, `_educators`, `_conclusions` — plus `ai_refs`), so
   the deck went from 1 live section ("AI Drives") to 7. `llm_promptengg_educators.tex` had never
   been compiled anywhere before (the other 4 files were already proven via other live decks), so
   this one got a verification compile: `Main_Seminar_AI_For_Educators_Presentation.tex` → 90
   pages, 0 fatal errors, only the usual harmless per-frame-navigation overfull warning. Build
   leftovers cleaned.
3. **SeqSeg — DONE, retired.** Repo-wide sweep before moving found an extra stray file not
   previously known: `Main_Seminar_LLM_SeqSeq_CheatSheet.tex` (typo "SeqSeq" vs "SeqSeg"), also
   `\input`-ing `seminar_seqseg_content.tex`, with no matching Presentation driver — moved along
   with everything else. Moved to `_retired/llm_seqseg/`: both real `Main_Seminar_LLM_SeqSeg_*`
   drivers, the stray typo'd CheatSheet, `seminar_seqseg_content.tex`, and the 3 now-orphaned raw
   topic files (`seq_intro.tex`, `seqseg_intro.tex`, `seqseg_refs.tex` — none had any other
   consumer left after the wrapper moved). Dropped the dead link from `COURSES.md`'s "Applied ML"
   bullet. Confirmed zero remaining references repo-wide.
4. ~~Recompile remaining 27 Cluster-1 drivers~~ — **DECIDED: no, skip.** Matches the established
   "pure renames don't need recompiling" precedent.
5. **Neo4j/DataScience `_short_content` rename — DONE.** `seminar_graph_neo4j_content.tex` →
   `seminar_graph_neo4j_short_content.tex`, `seminar_graph_ds_content.tex` →
   `seminar_graph_ds_short_content.tex`; all 4 driver `\input` lines
   (`Main_Seminar_Graph_Neo4j_*`, `Main_Seminar_Graph_DataScience_*`, Presentation + CheatSheet
   each) updated. Confirmed zero remaining references to the old names. Not recompiled — cosmetic
   rename only, no content change.
6. **Phase 5 content-skeleton pass — HELD, not executed.** Read all 7 remaining content-aggregator
   files in the family before deciding. 3 of them (`ai_intro_biz_leaders.tex` 52 frames,
   `ai_intro_tech_leaders.tex` 76 frames, `ai_intro_project_managers.tex` 41 frames — the
   BizLeaders/TechLeaders/ProjectManagers decks) are already rich, carefully-authored single-file
   decks with no `\section` markup at all, but a clear implicit narrative arc (intro → concepts →
   applications/hands-on → risk/governance → roadmap/closing) already present in the frame
   sequence. Imposing the explicit skeleton on these means inserting `\section` breaks into
   someone else's already-good flow — exactly the class of edit this repo's own conventions say
   needs page-by-page visual verification (silent overflow/duplication bugs don't show up in the
   compile log; this session hit that twice already, with the SeqSeg duplicate and the Session 7/8/9
   overflow findings referenced elsewhere in `CLAUDE.md`). Judged that real per-deck editorial
   work across all ~10 decks in the family is too large and too risky to do unsupervised in one
   pass without visual checks between decks — chose to hold rather than either skip silently or
   risk degrading already-good content. **Next step when resumed**: go deck-by-deck (suggest
   starting with the thinner ones — Kids, WithML, Career Short — before the 3 large single-file
   decks), rendering pages to images after each edit per the repo's established verification
   discipline, not a single blind pass across all 10.

---

Approved plan (Aug 2026). Goal: fold the three currently-separate-but-already-content-linked
families of talks — "AI for audience X" (`Main_Seminar_AI_*`), "ChatGPT/LLM intro"
(`Main_Seminar_LLM_ChatGPT_*`), and "Career in Data Science" (`Main_Seminar_Tech_
CareerInDataScience*`) — into one family under a single naming convention:
`Main_Seminar_AI_<Topic>_<Audience>_{Presentation,CheatSheet}.tex`, Topic in
{Overview, ChatGPT, Career}.

Rename scope: `Main_Seminar_*` driver files + their 1:1 content-aggregator wrapper files only.
Do NOT rename the shared topic files underneath (`ai_intro_tech.tex`, `career_ai_midcareer.tex`,
etc.) — those are consumed by other decks too; only their `\input` sites inside the renamed
wrappers change path.

Tool-demo seminars (`AI_OpenCode`, `AI_Sarvam`, `AI_ClaudeCode`) are a different genre
(hands-on walkthroughs, not audience-tiered talks) — out of scope, no rename.

## Phase 1 — AI Overview sub-family — DONE (Aug 2026)
- [x] 1.1 Grep repo-wide for consumers of each content-aggregator file below before renaming
      (`seminar_artificialintelligence_{tech,nontech,kids,educators}_content.tex`,
      `seminar_artificialintelligence_{biz_leaders,tech_leaders,project_managers}.tex`,
      `seminar_artificialintelligencemachinelearning_content.tex`) — confirm each is 1:1 with
      its driver, not shared elsewhere, before moving it. Found one non-1:1 case:
      `seminar_artificialintelligence_tech_content.tex` is also `\input` by `workshop_ai_content.tex`
      — updated alongside the rename, not just the driver.
- [x] 1.2 Renamed driver pairs + their content-aggregator file (content files also renamed to
      `seminar_ai_overview_<sub>_content.tex` for naming-convention consistency, including fixing
      3 files that had never carried the `_content` suffix — biz_leaders/tech_leaders/project_managers):
      - `Main_Seminar_AI_{Presentation,CheatSheet}` + `seminar_artificialintelligence_tech_content.tex` → `AI_Overview_General` + `seminar_ai_overview_general_content.tex`
      - `Main_Seminar_AI_for_NonTech_*` + `seminar_artificialintelligence_nontech_content.tex` → `AI_Overview_NonTech` + `seminar_ai_overview_nontech_content.tex`
      - `Main_Seminar_AI_for_Kids_*` + `seminar_artificialintelligence_kids_content.tex` → `AI_Overview_Kids` + `seminar_ai_overview_kids_content.tex`
      - `Main_Seminar_AI_for_Educators_*` + `seminar_artificialintelligence_educators_content.tex` → `AI_Overview_Educators` + `seminar_ai_overview_educators_content.tex`
      - `Main_Seminar_AI_BizLeaders_*` + `seminar_artificialintelligence_biz_leaders.tex` → `AI_Overview_BizLeaders` + `seminar_ai_overview_bizleaders_content.tex`
      - `Main_Seminar_AI_TechLeaders_*` + `seminar_artificialintelligence_tech_leaders.tex` → `AI_Overview_TechLeaders` + `seminar_ai_overview_techleaders_content.tex`
      - `Main_Seminar_AI_ProjectManagers_*` + `seminar_artificialintelligence_project_managers.tex` → `AI_Overview_ProjectManagers` + `seminar_ai_overview_projectmanagers_content.tex`
- [x] 1.3 DECIDED (user, Aug 2026) — `Main_Seminar_AI-ML_*` renamed to `AI_Overview_WithML`,
      kept standalone (not folded into `AI_Overview_General`). Content file renamed
      `seminar_artificialintelligencemachinelearning_content.tex` → `seminar_ai_overview_withml_content.tex`.
- [ ] 1.4 Content gap flag (not a rename task, still open): `AI_Overview_Educators` has 5 of 6
      planned sections commented out (only "AI Drives" live). Decide later whether to revive via a
      dedicated content pass.

## Phase 2 — ChatGPT sub-family — DONE (Aug 2026)
- [x] 2.1 Renamed driver pairs + content-aggregator file (content files renamed to
      `seminar_ai_chatgpt_<sub>_content.tex`):
      - `Main_Seminar_LLM_ChatGPT_{Presentation,CheatSheet}` + `seminar_chatgpt_content.tex` → `AI_ChatGPT_General` + `seminar_ai_chatgpt_general_content.tex`
      - `Main_Seminar_LLM_ChatGPT_FromZero_*` + `seminar_chatgpt_fromzero_content.tex` → `AI_ChatGPT_FromZero` + `seminar_ai_chatgpt_fromzero_content.tex`
      - `Main_Seminar_LLM_ChatGPT_FromZeroShort_*` + `seminar_chatgpt_fromzeroshort_content.tex` → `AI_ChatGPT_FromZeroShort` + `seminar_ai_chatgpt_fromzeroshort_content.tex`
      - `Main_Seminar_LLM_ChatGPT_NonTech_*` + `seminar_chatgpt_nontech_content.tex` → `AI_ChatGPT_NonTech` + `seminar_ai_chatgpt_nontech_content.tex`
      - `Main_Seminar_LLM_ChatGPT_TechShort_*` + `seminar_chatgpt_techshort_content.tex` → `AI_ChatGPT_TechShort` + `seminar_ai_chatgpt_techshort_content.tex`
      - `Main_Seminar_LLM_ChatGPT_Mech_*` + `seminar_chatgpt_mech_content.tex` → `AI_ChatGPT_Mech` + `seminar_ai_chatgpt_mech_content.tex`
      Two non-1:1 cases found and fixed: `seminar_chatgpt_content.tex` is also nested by
      `seminar_chatgpt_fromzero_content.tex` (its `\input` updated to the new name); and
      `seminar_chatgpt_fromzeroshort_content.tex` is also `\input` by `workshop_llm_content.tex`
      (also updated).
- [x] 2.2 DECIDED (user, Aug 2026) — leave `FromZero`'s full nesting of `General` via `\input` as
      intentional/documented, not converted to the comment-sibling pattern.

## Phase 3 — Career sub-family — DONE (Aug 2026)
- [x] 3.1 Renamed driver pairs + content-aggregator file:
      - `Main_Seminar_Tech_CareerInDataScience_*` + `seminar_careerindatascience_content.tex` → `AI_Career_Full` + `seminar_ai_career_full_content.tex`
      - `Main_Seminar_Tech_CareerInDataScience_Short_*` + `seminar_careerindatascience_short_content.tex` → `AI_Career_Short` + `seminar_ai_career_short_content.tex`
      `seminar_careerindatascience_content.tex` is also `\input` by `workshop_ai_content.tex` —
      updated alongside the rename.

## Phase 4 — repo-wide reference updates (after Phases 1-3 land) — MOSTLY DONE (Aug 2026)
- [x] 4.1 Updated `COURSES.md` (driver links + the "AI overviews", "ChatGPT / LLM intros",
      "Career & meta" bullet lists, plus one incidental `Main_Seminar_LLM_ChatGPT_FromZeroShort`
      link inside the LLMs workshop bullet) to the new names.
- [x] 4.2 Checked `README.md` — does not name any of these decks directly, nothing to change.
- [x] 4.3 Repo-wide grep for the old filenames (`.tex`, `.md`, `.bat`) after all renames: zero
      remaining references except this todo file itself (expected, it documents the old names).
- [~] 4.4 Recompile every renamed driver (`texify -cp`) — **partially done, deliberately stopped
      early.** 5 of 32 compiled clean (`AI_Overview_{BizLeaders,Educators,General,Kids,NonTech}
      _Presentation`) before the run was aborted per user judgment call: since this phase only
      renamed files and fixed up `\input` paths (no content edits), a full 32-driver recompile
      was deemed low-value verification. Remaining 27 drivers not recompiled. Their throwaway
      PDFs from the partial run were deleted (not repo deliverables).
- [x] 4.5 Clean build leftovers — n/a beyond the above; `texify -cp` leaves no `.aux/.log/...`
      behind on success, and no stray files were left by the aborted run.

## Phase 5 — deferred: common section-skeleton content pass
Apply `Background(opt) → Introduction → How it Works/Fluency(opt) → Applications/Impact →
Demo(opt) → Career/Getting Involved(opt) → Conclusion → References` uniformly across the
renamed family. This is a content-level `/upgrade-deck`-style pass, done only after the
renames land and are verified — separate task, not started yet.

---

## Cluster 2 — Graph/LLM/NLP/ML seminars — analyzed and resolved (Aug 2026)
`\input`-chain trace completed for all 13 standalone Graph/LLM/NLP/ML seminar drivers spotted
from `COURSES.md`'s "Other Notable Standalone Seminars" list (11 originally named + 2 more found
while tracing: Graph GeometricDeepLearning, Graph NeuralNetworks) against every existing
Graph/LLM/NLP/ML workshop's `\input` chain. Finding: this is **not** a uniform "fold seminar into
workshop" job — six different situations came up. Nothing below has been executed; each bucket
needs its own go/no-go before any file is touched.

**A. Already folded, no action needed**
- Graph NeuralNetworks (`seminar_graph_gnn_content.tex`, standalone driver
  `Main_Seminar_Graph_NeuralNetworks_*`) is the exact same content file already `\input` as the
  "GNN" section of `workshop_graph_gdl_content.tex` (Geometric Deep Learning Workshop). The
  standalone seminar and the workshop section are already one and the same file — nothing to do.

**B. Clean fold candidates — real content gap in an existing workshop — DONE (Aug 2026)**
- Graph RAG and SQL+RAG folded into `workshop_llm_rag_content.tex` (LLM RAG Workshop), which had
  a real gap (its "Advanced RAG" seminar covers Multimodal/Production/VertexAI but nothing graph-
  or SQL-based). Folded via the raw topic files directly (`graph_rag_intro`/`_impl_microsoft`/
  `_conclusion`/`_refs`, `sql_rag_intro`/`_concepts`/`_impl_llamaindex`/`_impl_vanna`/
  `_conclusions`/`_refs`), **not** by `\input`-ing `seminar_graph_rag_content.tex` wholesale —
  that wrapper opens with `\input{llm_rag_intro}`, the same file `seminar_llm_rag_content.tex`
  (already first in this workshop) already brings in, so including the wrapper as-is would have
  duplicated the RAG-intro section. Standalone `Main_Seminar_Graph_RAG_*`/`Main_Seminar_LLM_SQL_RAG_*`
  drivers are untouched and still work on their own. Not recompiled — the folded-in topic files
  are unmodified and already compile clean under their own standalone drivers, so this is treated
  as the same "no compile needed" case as a pure rename.

**C. Redundant — retired (Aug 2026)**
- Text Mining (`seminar_textmining_content.tex`: `nlp_infoextract`, `nlp_topicmodeling`,
  `nlp_ner`) duplicated raw topic files the NLP Workshop already pulls in more deeply:
  `seminar_nlp_pos_ner_content.tex` has `nlp_ner` *plus* `nlp_ner_nltk`/`_spacy`/`_crf`/`_lstm`;
  `seminar_nlp_advanced_content.tex` has `nlp_topicmodeling` *plus* `nlp_topicmodeling_gensim`,
  and `nlp_infoextract`. Deleted `Main_Seminar_NLP_TextMining_{Presentation,CheatSheet}.tex` and
  `seminar_textmining_content.tex` (the underlying raw topic files stay — they're still used by
  the NLP Workshop). Removed the Text Mining link from `COURSES.md`'s "Applied ML" bullet.
  No other file referenced it. Not recompiled — nothing was folded in, only removed.

**D. Overlapping pair — investigated (Aug 2026), turned out to be a non-issue**
- Graph+NLP (`seminar_graph_nlp_content.tex`: `graph_intro`, `graph_gnn_intro`, `graph_nlp`)
  shares 2 of its 3 `\input` targets with Graph NeuralNetworks (`seminar_graph_gnn_content.tex`:
  `graph_intro`, `graph_gnn_intro`, `graph_graphtransformer_intro`, `graph_gnn_refs`). Checked
  whether the two remaining files were also near-duplicates despite both having exactly 36 frames
  — `diff graph_nlp.tex graph_graphtransformer_intro.tex` shows they are completely different,
  unrelated content (NLP-application-of-GNNs slides borrowed from a public tutorial, vs a
  well-authored from-scratch Graph Transformer explainer; the 36/36 frame-count match was
  coincidence). So the only real overlap is the two shared intro files, and that's the same
  "multiple decks share a common foundational topic file" pattern already used repo-wide (e.g.
  `ai_intro_tech.tex` shared across 5 decks) — not a bug, not worth merging. **No action needed.**

**E. Already has a working short/full split — no fold needed, optional rename only**
- Graph Neo4j (`seminar_graph_neo4j_content.tex`, standalone) vs `seminar_graph_neo4j_full_content.tex`
  (used by `workshop_graph_db_content.tex`, Graph Database Workshop) — the full version adds
  `graph_neo4j_concepts`, `graph_cypher`, `graph_python`.
- Graph DataScience (`seminar_graph_ds_content.tex`, standalone) vs
  `seminar_graph_datascience_full_content.tex` (same workshop) — the full version adds
  `graph_datascience_neo4j`, `graph_certification`.
  Both pairs already coexist correctly as a short-standalone/full-in-workshop split, the same
  shape as the CareerInDataScience Full/Short precedent, just with the suffix on the *full* file
  instead of the *short* one. No content action needed. Optional cosmetic cleanup: rename
  `seminar_graph_neo4j_content.tex`/`seminar_graph_ds_content.tex` to end in `_short_content` to
  match the repo's usual `X`/`X_short` naming direction — low priority, naming-only.

**F. Partially subsumed by an existing workshop — needs a content decision**
- Knowledge Graphs has two standalone seminars (`Main_Seminar_Graph_KnowledgeGraphs_*` →
  `seminar_kg_content.tex`: `kg_overview`, `kg_conclusions`, `graph_refs`; and
  `Main_Seminar_LLM_KnowledgeGraphs_*` → `seminar_llm_kg_content.tex`: `kg_llm_intro`,
  `kg_llm_conclusions`, `kg_llm_refs`) plus a separate, much larger standalone Knowledge Graph
  **workshop** (`Main_Workshop_Graph_KnowledgeGraph_*` → `workshop_graph_kg_content.tex`:
  `graph_intro`, `graph_algorithms`, `kg_intro`, `kg_semantics`, `kg_llm_intro`,
  `kg_llm_conclusions`, `kg_implementations`, `kg_conclusions`, `graph_refs`).
  - LLM_KnowledgeGraphs was effectively already subsumed: the workshop already had
    `kg_llm_intro`+`kg_llm_conclusions`, missing only `kg_llm_refs`.
  - Graph_KnowledgeGraphs: `kg_overview.tex` vs `kg_intro.tex` (the workshop's own intro file)
    checked with `diff` — byte-identical for the first 326 of `kg_intro.tex`'s 405 lines, i.e.
    `kg_overview.tex` is a strict superset. Its extra ~150 lines ("KG with Neo4j"
    Introduction/Framework/Workflows/Architecture + generic Conclusion/References) read as
    unfinished placeholder text, not real authored content — the References frame literally says
    *"List of references, papers, and resources used in this presentation. Include links, books,
    articles, and other relevant sources."*, a template instruction never filled in. Nothing in
    `kg_overview.tex` was worth grafting into the workshop.
  - **DECIDED (user, Aug 2026) — retired both.** Added `\input{kg_llm_refs}` to
    `workshop_graph_kg_content.tex`'s "With LLMs" section first (so that reference list isn't
    lost), then deleted `Main_Seminar_Graph_KnowledgeGraphs_{Presentation,CheatSheet}.tex`,
    `seminar_kg_content.tex`, `Main_Seminar_LLM_KnowledgeGraphs_{Presentation,CheatSheet}.tex`,
    and `seminar_llm_kg_content.tex`. Updated `COURSES.md`'s "Graph topics" bullet to drop both
    dead links (the Knowledge Graph Workshop was already listed separately in the workshops
    table, so no new link needed). No other file referenced any of the 6 removed files. Not
    recompiled — same "no compile needed" reasoning as bucket B (nothing new was authored, only
    removed/relocated a reference).
  - Note: `workshop_graph_gdl_content.tex` already carries a comment recording that the repo
    author considered and rejected pulling Knowledge Graphs into the Geometric Deep Learning
    Workshop ("never authored at seminar scope; only a full standalone workshop exists ... too
    large a fit for one section here") — so KG's natural home is confirmed to be its own
    workshop, not GDL.

**G. No existing workshop home — not a mechanical fold, needs a bigger decision**
- Seq2Seq (`seminar_seqseg_content.tex`) — no LLM or time-series workshop exists anywhere in the
  repo to fold it into. Also thin as authored: 13 lines, and its "Approaches" section is a bare
  `\section` header with no `\input` under it at all.
  - **Separate real bug found while sizing this up (Aug 2026), fixed, but only partially —
    a deeper problem remains open.** `seq_intro.tex` and `seqseg_intro.tex` (914 lines / 73
    frames each) were **byte-identical** (`diff` showed zero difference), and
    `seminar_seqseg_content.tex` `\input` *both* under two different section headers
    ("Introduction to Sequences/Time-Series" and "Sequence Segmentation") — rendering the same 73
    frames twice. **DECIDED (user, Aug 2026) — drop the duplicate only, for now**: removed the
    `\section[SeqSeg]{Sequence Segmentation}` + `\input{seqseg_intro}` pair from
    `seminar_seqseg_content.tex`, leaving a comment explaining why and pointing at the unresolved
    issue below. **Still open, deliberately not fixed this pass**: `seq_intro.tex` itself is
    misfiled content — its frame titles ("NLP is AI", "The Promise of NLP", "NLP and
    Intelligence", "Recent NLP Applications") and its references file (NLTK, Named Entity
    Recognition, Topic Modeling, Word2Vec) are generic NLP-101 material, not Seq2Seq models or
    time-series segmentation. So `Main_Seminar_LLM_SeqSeg_*` no longer repeats itself, but still
    shows content that doesn't match its own title — needs a real Seq2Seq content-authoring pass,
    out of scope for this consolidation. Not recompiled, per the "pure removal, no compile needed"
    reasoning used throughout this cluster.
- Explainable AI (`seminar_explainableai_content.tex`: 4 topic files, 11+17+17+5+7 = 57 frames
  total across `ai_seminar_xai_{overview,icertis,aiintro,contrary,conclusion}.tex`) and Matrix
  Profile (`seminar_matrixprofile_content.tex`: 4 topic files, 17+15+28+7 = 67 frames total) are
  both substantial, fully-authored standalone decks (not thin stubs) — grouped with Reinforcement
  Learning on `COURSES.md`'s "Applied ML" line, but RL already has its own **full standalone
  workshop** (`Main_Workshop_ML_ReinforcementLearning_*`) rather than being folded into any of the
  6 ML-course workshops (Foundations/Regression/TreeBased/SupervisedII/Unsupervised) — none of
  which references either topic today. **DECIDED (user, Aug 2026) — leave both standalone**,
  matching the RL precedent; their size is comparable to RL's own content, not a thin stub needing
  a home. No files touched.

**Cluster 2 status: all 7 buckets resolved (Aug 2026).** A, D: no action needed (non-issues).
B: Graph RAG + SQL RAG folded into the LLM RAG Workshop. C: Text Mining retired. E: no action
needed (optional rename only, left for later). F: both standalone Knowledge Graph seminars
retired. G: Explainable AI/Matrix Profile left standalone (matches the RL precedent); SeqSeg's
duplicate-`\input` bug fixed, but the deeper "wrong content under the Seq2Seq title" problem is
still open for a future dedicated content-authoring pass on `seq_intro.tex`.

## Cluster 3 — Tech and Data small clusters — analyzed and resolved (Aug 2026)
Prompted by the user asking to check a small "Tech" cluster and a small "Data" cluster for the
same kind of redundancy Cluster 2 found.

**Data cluster (`Main_Seminar_Data_*`)** — no issue, already a healthy pattern, no action:
- `Main_Seminar_Data_Analytics` (`seminar_dataanalytics_content.tex`: `data_intro_short`,
  `data_conclusion`, `data_refs_short`) is the intentional "short" 1-hour standalone sitting
  alongside the much larger Data Analytics Workshop (`workshop_dataanalytics_content.tex`:
  `data_intro` full + dimensionality/pandas/prep/exploration/viz, sharing `data_conclusion`) —
  same short/full split shape as Cluster 2 bucket E (Graph Neo4j/DataScience).
- `Main_Seminar_Data_Concepts_Overview` (`seminar_dataconcepts_overview_content.tex`:
  `data_intro_overview`) is the already-documented "overview" deep-dive precedent from an earlier
  session (see the "Short/full topic-file sync audit" note in `CLAUDE.md`).

**Tech cluster (`Main_Seminar_Tech_*`)**:
- HypeCycles_Gartner (`seminar_hypecyles_gartner_content.tex`, 942 lines) and LaTeX_Research
  (`seminar_latex4research_conent.tex`, 896 lines, filename typo pre-existing/documented) are
  both substantial, fully standalone, unique topics with no overlap anywhere else — no action.
- **Mentoring — retired (Aug 2026).** Its unique content (`seminar_mentorship_content.tex`) was a
  PAIC (Pune AI Community) mentee-project list: Ask Yogasutra, MidcurveNN, Mining Resume,
  Sarvadnya RAG Systems, Nature of Code — not duplicated anywhere else in the repo, so not
  "redundant" in the Cluster 2/Text-Mining sense. But at least one entry was confirmed stale: this
  repo's own `GitHub/CLAUDE.md` records MidcurveNN as closed on 2026-08-13 (remaining work filed
  as GitHub issues, next phase promoted to a written research topic) — no longer an open mentee
  project the way the deck described it. Could not verify the other 4 projects' current status
  (external repos outside available access). User call: retire the whole deck regardless of
  per-project staleness, not just the stale MidcurveNN entry. Deleted
  `Main_Seminar_Tech_Mentoring_{Presentation,CheatSheet}.tex` and `seminar_mentorship_content.tex`.
  `paic.tex` (the PAIC contact/QR-code slide, also `\input` by this deck) was **kept** — it's
  shared with 8 other decks including `Main_Seminar_AI_Career_Full`, so only the deck-specific
  project list went. Updated `COURSES.md`'s "Career & meta" bullet to drop the dead Mentoring
  link. No other file referenced the removed files. Not recompiled — pure removal, same reasoning
  as bucket C/F above.

## Retirement convention changed: move to `_retired/`, don't delete (Aug 2026)
Starting partway through Cluster 4 below, the user asked that retired files be moved into a
`LaTeX/_retired/<topic>/` folder rather than `rm`-deleted, so the content stays recoverable
without needing git. Applied to GCP (`_retired/gcp/`), the 6 retired ChatGPT drivers+content
(`_retired/ai_chatgpt/`), and Sarvam (`_retired/ai_sarvam/`). **Not applied retroactively**: Text
Mining, both Knowledge Graph seminars, and Mentoring (all deleted earlier in Cluster 2/3, before
this convention existed) were hard-deleted with `rm` and are only recoverable via the user's own
git client (nothing was committed at deletion time) — flagged to the user, not fixed here since
recovering them requires git commands this session cannot run.

## Cluster 4 — GCP retirement + AI_Overview/AI_ChatGPT restructure + tool-demo cleanup (Aug 2026)
Prompted by the user reviewing `Main_Seminar_AI_*` directly and disliking the two-different-topic-
word naming (`AI_Overview_X` vs `AI_ChatGPT_X`) left over from Cluster 1.

**GCP cluster retired in full.** Traced all 4 GCP standalone seminars
(`Main_Seminar_GCP_{,GenAI,VertexAI,DocAI}_*`). `genai_intro.tex` (the one substantial, genuinely
reusable file `seminar_gcp_genai_content.tex` pulled in, 81 frames) was already shared with
`seminar_llm_genai_promptengg_content.tex` — nothing needed repurposing, it already lives on
independently. The rest (`gcp_overview.tex`, `gcp_vertexai_overview.tex`,
`gcp_docai_overview.tex`) turned out to be the same generic AI-template-boilerplate pattern found
in Cluster 2's `kg_overview.tex` finding — read `gcp_vertexai_overview.tex` in full to confirm:
generic, unspecific prose with no real product detail, no code, no citations. `gcp_genai_intro.tex`
(27 frames) was genuinely authored with real citations/screenshots but is GCP/Vertex-AI-specific
with no other natural home. Per user instruction ("if not [repurposable], remove it fully"): moved
all 4 driver pairs, all 4 content-aggregator files, and the 5 GCP-exclusive raw topic files
(`gcp_overview`, `gcp_refs`, `gcp_genai_intro`, `gcp_vertexai_overview`, `gcp_docai_overview`) to
`_retired/gcp/`. `genai_intro.tex`/`genai_refs.tex` left in place (shared elsewhere). Removed the
"Google Cloud Platform" bullet from `COURSES.md`. Confirmed zero other references repo-wide before
moving.

**AI_Overview_*/AI_ChatGPT_* unified into one `AI_For_<X>` pattern**, per explicit user direction
("AI for All Tech... AI for All non-Tech... AI for Educators... even those chatgpt included tex
files can be folded into these common pattern seminars"):
- 6 audience-only Overview decks renamed with no content change (Kids, Educators, BizLeaders,
  TechLeaders, ProjectManagers, WithML) → `Main_Seminar_AI_For_<Audience>_*` +
  `seminar_ai_for_<audience>_content.tex`.
- `AI_Overview_General` → **`Main_Seminar_AI_For_All_Tech`**, `AI_Overview_NonTech` →
  **`Main_Seminar_AI_For_All_NonTech`** (the two names the user gave explicitly), each also
  **absorbing its matching ChatGPT sub-family content** as new sections, per the user's explicit
  choice to fold all 4 non-audience ChatGPT variants (FromZero/FromZeroShort/TechShort/Mech) into
  `AI_For_All_Tech` rather than keep them as separate drivers.
  - **Dedup check done before folding** (same discipline as Cluster 2 bucket B): confirmed via
    `diff` that `llm_fromzero_short.tex`/`chatgpt_intro_short.tex` are the exact same source as
    `llm_fromzero.tex`/`chatgpt_intro.tex` with frames commented out (the repo's standard
    `X`/`X_short` sibling pattern) — not independent content. So `AI_For_All_Tech`'s merged content
    (`seminar_ai_for_all_tech_content.tex`) uses the **full** versions once
    (`llm_fromzero`, `chatgpt_intro`) plus `chatgpt_applications`, `chatgpt_applications_imi`
    (Mech's unique addition), `llm_promptengg_sandwich` (TechShort's unique demo), and
    `chatgpt_conclusion` (shared by all 4, included once) — never the `_short` siblings, avoiding
    the kind of literal duplication found and fixed in the SeqSeg bug above.
  - `AI_For_All_NonTech`'s merge (`seminar_ai_for_all_nontech_content.tex`) deliberately keeps
    `chatgpt_intro_short` (not the full version) — the simplified telling is the right choice for
    this audience — plus `chatgpt_applications`, `llm_promptengg_sandwich`, `career_ai_midcareer`,
    and `chatgpt_conclusion`, from the retired `ChatGPT_NonTech` driver.
  - Confirmed no content collision before folding: `seminar_llm_intro_content.tex` (the separate
    "LLM" section other decks pull in) uses entirely different files (`llm_intro`/`llm_conclusion`/
    `llm_refs`), so the fold adds no duplicate frames anywhere downstream.
- **All 6 ChatGPT driver pairs retired** (General, NonTech, FromZero, FromZeroShort, TechShort,
  Mech) + their 6 content-aggregator files → moved to `_retired/ai_chatgpt/`. Their content lives
  on either folded into `AI_For_All_Tech`/`AI_For_All_NonTech`, or (FromZeroShort specifically)
  inlined directly into `workshop_llm_content.tex` (see below) — nothing was lost, only the
  standalone driver layer was removed.
- **Fixed 2 real dangling references found during the repo-wide sweep**:
  `workshop_ai_content.tex`'s "AI" section pointed at the old `seminar_ai_overview_general_content`
  — repointed to `seminar_ai_for_all_tech_content` (confirmed no duplicate-content risk with that
  workshop's separate LLM section, so its "AI" section is now richer, not broken, as a side effect
  of the fold — left as-is rather than fought). `workshop_llm_content.tex` pointed at the
  now-retired `seminar_ai_chatgpt_fromzeroshort_content` — since that content is FromZeroShort's
  original 3 raw `\input` lines (`llm_fromzero_short`, `chatgpt_intro_short`, `chatgpt_conclusion`),
  inlined those 3 lines directly in place of the dead reference, so this workshop's content is
  byte-for-byte unchanged despite the wrapper's retirement.
- `COURSES.md`'s "AI overviews"/"ChatGPT / LLM intros" bullets merged into one "AI for different
  audiences" bullet reflecting the new names; the LLM Workshop bullet's dead
  `AI_ChatGPT_FromZeroShort` link changed to plain text ("workshop-only", matching the existing
  "LLM applications (workshop-only)" convention already used on that same line).
- Not recompiled — same "fold in already-verified content, no compile needed" reasoning as
  Cluster 2 bucket B, now also covering the mechanical renames.

**Tool-demo trio resolved**: `AI_Sarvam` retired in full (driver pair + `ai_tools_sarvam_intro.tex`
→ `_retired/ai_sarvam/`; confirmed zero other references, it was never listed in `COURSES.md`).
`AI_ClaudeCode`/`AI_OpenCode` **kept** (per user: "we have only two hands-on workshops... they will
remain") and renamed to `Main_Seminar_AI_HandsOn_ClaudeCode_*`/`Main_Seminar_AI_HandsOn_OpenCode_*`
— content-aggregator filenames (`seminar_ai_claudecode_content.tex`, `ai_tools_opencode_intro.tex`)
left unchanged, only the driver-level segment was renamed. Fixed the 2 stale references this
surfaced: `CLAUDE.md`'s example `texify -cp` command and its "Known issues" note (both updated to
the new name; the note also flags the rename inline for future readers), and `COURSES.md`'s
"Career & meta" bullet (Claude Code never really belonged there) — split into its own new
"Hands-on tool workshops" bullet listing both ClaudeCode and OpenCode (OpenCode had never been
listed in `COURSES.md` before). Left `.claude/plans/upgrade-claudecode-deck.md` untouched — a
historical plan-file record, not a live reference.

**Remaining open items across this whole file**: Phase 1.4 (now `AI_For_Educators` — still has 5
of 6 sections commented out), Phase 4.4 (27 of 32 Cluster 1 drivers never recompiled, deemed low
priority), Phase 5 (deferred content-skeleton pass, scope now includes the Cluster 4 renames too),
bucket E's optional Neo4j/DataScience rename, the SeqSeg content-authoring gap, and recovering
the pre-`_retired/`-convention deletions (Text Mining, both Knowledge Graph seminars, Mentoring)
from git if the user wants them preserved too.
