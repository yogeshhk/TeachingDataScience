# LaTeX Hierarchy Analysis Report
_Generated: 2026-04-29 — **ALL 5 PHASES COMPLETED 2026-04-29**_

## Execution Summary

| Phase | Description | Result |
|-------|-------------|--------|
| Phase 1 | Claude Code content aggregator | DONE — created `seminar_ai_claudecode_content.tex`; updated both drivers |
| Phase 2 | 29 missing seminar CheatSheets | DONE — all 29 `Main_Seminar_*_CheatSheet.tex` created |
| Phase 3 | 10 missing workshop CheatSheets | DONE — all 10 `Main_Workshop_*_CheatSheet.tex` created |
| Phase 4 | Workshop → seminar-layer restructuring | DONE — 11 new seminar content files; 5 workshop files rewritten |
| Phase 5 | Course → workshop-layer restructuring | DONE — 1 new workshop content file; all 3 course files rewritten |

**Key Phase 4 decisions:**
- Created 11 new `seminar_*_content.tex` files to bridge gaps (DL foundations/TF/NLP basics/POS-NER/NLP-ML/NLP-advanced/graph variants/LLM applications/RAG-advanced)
- `seminar_nlp_ml_content.tex` creation also fixed a missing-file bug (referenced by `Main_Seminar_NLP_ML_Presentation.tex`)
- `seminar_deeplearning_tensorflow_core_content.tex` omits conclusion/refs to avoid duplicates (PyTorch seminar supplies them)
- `workshop_python_content.tex` and `workshop_maths4ml_content.tex` **deferred** (seminar_python_content covers only overview/DSA/sysdesign — insufficient; maths4ml mostly commented out)

**Key Phase 5 decisions:**
- `workshop_machinelearning_content.tex` **created** (no prior workshop existed for classical ML)
- `course_deeplearning_content.tex` retains 3 pre-req raw files (ai_intro_tech_short, python_intro_short, python_syntax_short) before the workshop input
- `course_generativeai_content.tex` reduced from 258 lines of raw files to 3 workshop references + 43 course-specific additions

**Known issues (not fixed — out of scope):**
- `seminar_latex4research_conent.tex` — filename has a typo (`conent` not `content`); both the file and its references would need renaming together

---

## Current Inventory (198 driver files total)

| Type | Total | Paired (P+C) | Presentation-only |
|------|-------|--------------|-------------------|
| Main_Seminar_* | 68 | 39 | 29 |
| Main_Workshop_* | 30 | 20 | 10 |
| Main_Course_* | 3 | 3 | 0 |

Plus: 69 `seminar_*_content.tex`, 33 `workshop_*_content.tex`, 3 `course_*_content.tex`

---

## Ideal Hierarchy

```
Course (40hr)    -> workshop_*_content.tex only
Workshop (4-16h) -> seminar_*_content.tex only (+ optional extras beyond seminar scope)
Seminar (1hr)    -> raw .tex topic files (via seminar_*_content.tex)
Driver files     -> Presentation.tex  (uses template_presentation)
                    CheatSheet.tex    (uses template_cheatsheet)
                    both share the same *_content.tex
```

---

## Gap Analysis

### GAP 1 — Claude Code: Missing content aggregator
- `Main_Seminar_AI_ClaudeCode_Presentation.tex` directly inputs `ai_tools_claudecode_intro`
- `Main_Seminar_AI_ClaudeCode_CheatSheet.tex` directly inputs `ai_tools_claudecode_demo`
- **No `seminar_ai_claudecode_content.tex` exists**
- `ai_tools_claudecowork.tex` exists on disk but is referenced nowhere
- Fix: create the content aggregator including all three raw files; update both drivers

### GAP 2 — 29 Seminars have no CheatSheet counterpart

**AI group (5):** AI-ML, AI, AI_for_Educators, AI_for_Kids, AI_for_NonTech
**Graph group (3):** Graph_DataScience, Graph_KnowledgeGraphs, Graph_Neo4j
**LLM group (19):** Agents, ChatGPT_FromZero, ChatGPT_FromZeroShort, ChatGPT_Mech,
  ChatGPT_NonTech, ChatGPT_TechShort, Evaluation, FineTuning, GenAI, GenAI_PromptEngg,
  Intro, KnowledgeGraphs, LlamaIndex, Production, PromptEngg, Reasoning, SQL_RAG, SeqSeg,
  Transformers
**Tech group (2):** LaTeX_Research, Mentoring

All 29 already have a `seminar_*_content.tex` — only the CheatSheet driver file is missing.

### GAP 3 — 10 Workshops have no CheatSheet counterpart

AI, LLM (main), LLM_Agents, LLM_Docling, LLM_LangChain, LLM_LangGraph,
LLM_RAG, LLM_Transformers, NLP_SpaCy, RAGToRiches

### GAP 4 — Workshop files bypass the seminar layer

| Workshop content file | Current pattern | Recommendation |
|----------------------|-----------------|----------------|
| `workshop_ai_content.tex` | **GOOD** — only seminar_* | No change |
| `workshop_llm_content.tex` | **MIXED** — seminar_ used for RAG/finetuning/agents/langchain/llamaindex/reasoning; ~15 raw files remain | Convert remaining raw files to existing seminar_llm_*_content references |
| `workshop_llm_rag_content.tex` | **MIXED** — 9 raw + `seminar_llm_langchain_content` | Replace core raw files with `seminar_llm_rag_content` |
| `workshop_deeplearning_content.tex` | **ALL RAW** (18 files) | Chain multiple DL seminar content files (DL, DL_TF, DL_PyTorch exist) |
| `workshop_naturallanguageprocessing_content.tex` | **ALL RAW** (30+ files) | Chain NLP seminar content files (NLP, NLP_Deep, NLP_ML, NLP_WordEmbeddings) |
| `workshop_python_content.tex` | **ALL RAW** (24 files) | Defer — seminar_python_content covers only overview/DSA/sysdesign (insufficient) |
| `workshop_graph_db_content.tex` | **ALL RAW** (11 files) | Route via seminar_graph_* content files |
| `workshop_maths4ml_content.tex` | **MOSTLY COMMENTED** (1 active) | Out of scope for now |

Note on DL workshop: `seminar_deeplearning_content.tex` contains only 3 raw files
(dl_intro_short, dl_intro_tensorflow, dl_refs_short). Multiple DL seminars exist
(DL_TF, DL_PyTorch, etc.) so the workshop should chain several seminar content files.

### GAP 5 — Course files bypass the workshop layer entirely

| Course content file | Current pattern | Fix |
|--------------------|-----------------|-----|
| `course_deeplearning_content.tex` | ALL RAW (22 files) | Replace with `\input{workshop_deeplearning_content}` |
| `course_machinelearning_content.tex` | ALL RAW (37 files) | Replace with relevant workshop_ml_* content files |
| `course_generativeai_content.tex` | ALL RAW (258 lines) | Replace with workshop_nlp + workshop_deepnlp + workshop_llm |

---

## Proposed Plan

_Changes to: seminar/workshop/course _content.tex files and Main_* drivers only.
No raw topic files are changed (exception: Claude Code demo+intro are merged via a new content file)._

```
Phase 1  Claude Code merge (3 files touched)
  1.1  CREATE  seminar_ai_claudecode_content.tex
               inputs: ai_tools_claudecode_intro, ai_tools_claudecode_demo, ai_tools_claudecowork
  1.2  EDIT    Main_Seminar_AI_ClaudeCode_Presentation.tex
               change \input{ai_tools_claudecode_intro} -> \input{seminar_ai_claudecode_content}
  1.3  EDIT    Main_Seminar_AI_ClaudeCode_CheatSheet.tex
               change \input{ai_tools_claudecode_demo}  -> \input{seminar_ai_claudecode_content}

Phase 2  Missing CheatSheet drivers for 29 seminars (29 new files)
  2.x  CREATE  Main_Seminar_X_CheatSheet.tex for each of the 29 (standard cheatsheet template)

Phase 3  Missing CheatSheet drivers for 10 workshops (10 new files)
  3.x  CREATE  Main_Workshop_X_CheatSheet.tex for each of the 10

Phase 4  Workshop restructuring (6 workshop_*_content.tex files edited)
  4a   EDIT    workshop_llm_content.tex — complete conversion to seminar layer
  4b   EDIT    workshop_llm_rag_content.tex — route core content via seminar_llm_rag_content
  4c   EDIT    workshop_deeplearning_content.tex — chain DL seminar content files
  4d   EDIT    workshop_naturallanguageprocessing_content.tex — chain NLP seminar content files
  4e   EDIT    workshop_graph_db_content.tex — route via seminar_graph_* content files

Phase 5  Course restructuring (3 course_*_content.tex files)
  5a   EDIT    course_deeplearning_content.tex -> workshop_deeplearning_content only
  5b   EDIT    course_machinelearning_content.tex -> relevant workshop_ml_* content
  5c   EDIT    course_generativeai_content.tex -> workshop_nlp + workshop_deepnlp + workshop_llm
```
