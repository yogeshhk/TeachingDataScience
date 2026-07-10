# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

An open-source educational repository containing:
- **LaTeX**: Beamer presentation slides and two-column cheatsheets for Data Science courses (Python, ML, DL, NLP, GenAI, RAG, Quantum Computing, etc.)
- **Code**: Python scripts and Jupyter notebooks demonstrating the concepts covered in the slides

## LaTeX Build System

Compile a specific deck from the `LaTeX/` directory using MikTeX's `texify`:
```bat
cd LaTeX
texify -cp Main_Seminar_AI_ClaudeCode_Presentation.tex
```

Compile all decks matching a pattern (Windows):
```bat
cd LaTeX
for /r %i in (Main_Seminar_*Educators*.tex) do texify -cp %i
```

Compile everything:
```bat
cd LaTeX
make_all.bat
```

## LaTeX Architecture

### 4-level content hierarchy

```
Course (40hr)     Main_Course_*_{Presentation,CheatSheet}.tex
                    └─ course_*_content.tex
                         └─ \input{workshop_*_content}  (+ course-specific extras)

Workshop (4-16hr) Main_Workshop_*_{Presentation,CheatSheet}.tex
                    └─ workshop_*_content.tex
                         └─ \input{seminar_*_content}  (seminar layer only)

Seminar (1hr)     Main_Seminar_*_{Presentation,CheatSheet}.tex
                    └─ seminar_*_content.tex
                         └─ \input{<domain>_<topic>}  (raw topic files)
```

Every deliverable has two output forms sharing the same content file:
- `Main_*_Presentation.tex` — Beamer slides (`\documentclass{beamer}`, uses `template_presentation.tex`)
- `Main_*_CheatSheet.tex` — Two-column landscape notes (`\documentclass{article}`, uses `template_cheatsheet.tex`)

CheatSheet column count convention: Seminars use `multicols{3}`; Workshops use `multicols{2}`.

`\usepackage{beamerarticle}` in `template_cheatsheet.tex` makes Beamer `\begin{frame}` environments compile correctly in article mode — no frame stripping needed.

### Naming conventions
- Topic files: `<domain>_<topic>.tex` (e.g., `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Driver files: `Main_[Course|Seminar|Workshop]_<Subject>_[Presentation|CheatSheet].tex`
  - Seminar ≈ 1 hour, Workshop ≈ 1 day, Course ≈ 1 week/semester
- Every Seminar and Workshop **must** have both a `_Presentation.tex` and a `_CheatSheet.tex` driver
  — this pairing rule holds even for the lighter-weight `_Overview` driver category below; only
  the "Seminar ≈ 1 hour" duration expectation doesn't apply to that category
- `_Short` suffix on a driver (e.g. `Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex`)
  denotes a shorter-duration variant of an existing seminar, sharing topic files with its
  parent via the `X.tex`/`X_short.tex` comment-sync pattern below — see the
  CareerInDataScience note for the first precedent of this at the seminar level
- `_overview` suffix on a topic file (e.g. `dl_intro_overview.tex`) denotes a deep/
  comprehensive standalone treatment that sits *outside* the `X.tex`/`X_short.tex`
  comment-sibling relationship — see the short/full sync audit note below
- `_Overview` suffix on a **driver** (e.g. `Main_Seminar_DL_Foundations_Overview_
  Presentation.tex`) denotes a minimal, single-section standalone seminar that exists
  solely to make an `_overview.tex` topic file independently reachable as its own
  session — deliberately thinner than a normal ~1hr Seminar (no References section,
  by design) — see the short/full sync audit note below for the 4 precedents

### Known issues
- `seminar_latex4research_conent.tex` — filename typo (`conent` vs `content`); the file and all references would need renaming together
- `seminar_quantum_content.tex` — orphaned; its former driver files (`Main_Seminar_Tech_Quantum_*`) were merged into `Main_Seminar_Tech_QuantumComputing_Overview_*`. Kept as reference; safe to delete.
- `Main_Seminar_AI_ClaudeCode_CheatSheet.tex` (only active content: `ai_tools_claudecode_demo_cadcam.tex`) walks through building `stlinspector`, paired with actual code at `Code/claudecode/CadCamWorkshop/` (untracked as of Jul 2026). As of Jul 2026 the deck and the PoC are back in sync: flat `src/` layout with no packaging (no `pyproject.toml`, no console-script entry point), the two-step `load_mesh`/`inspect_mesh` API, JSON-only reports (no Markdown report format), and a `thin_walls` check added alongside the original three. `CadCamWorkshop` now has both `.claude/skills/geometry-validation/` and `.claude/skills/inspection-report-summary/`; `.claude/agents/devops.md` was removed. `Code/claudecode/trial/` (also untracked) was a from-scratch dry run of the same workshop script, used to find and fix these drift points plus several missing/misplaced YAML-frontmatter fences in the tex's subagent/command/skill blocks — it's now redundant and pending manual deletion.
- `workshop_deepnlp_content.tex` line 2 has `\input{nlp_intro_short}` but no `nlp_intro_short.tex` exists in `LaTeX/` (closest matches: `nlp_intro_short_old.tex`, `nlp_intro_short_w_embedding.tex`, `nlp_intro.tex`) — blocks `Main_Workshop_NLP_Deep_*` and (via `course_generativeai_content.tex`) `Main_Course_GenerativeAI_*` from compiling. Found during the short/full sync audit below (Oct 2026); pre-existing and unrelated, not fixed.

### Quantum Computing course (added May 2026)
Full 4-level hierarchy for the course "Quantum Computing for Non-Physicists":
- **Course**: `Main_Course_QuantumComputing_{Presentation,CheatSheet}.tex` → `course_quantumcomputing_content.tex`
- **Workshops** (4): `Main_Workshop_QuantumComputing_{Foundations,Circuits,Algorithms,Advanced}_{Presentation,CheatSheet}.tex`
- **Seminar**: `Main_Seminar_Tech_QuantumComputing_Overview_{Presentation,CheatSheet}.tex` (~2 hr overview + QML)
- **Topic files** (14): all prefixed `quantum_` (e.g. `quantum_intro_motivation.tex`, `quantum_gates_circuits.tex`)
- Existing files `quantum_basics_intro.tex`, `quantum_maths_intro.tex`, `quantum_machinelearning_intro.tex` pre-date the course and are referenced optionally (heavier math; commented-out in the new overview seminar)

### Machine Learning course restructured (June 2026)
Full 4-level hierarchy for the 40-hour ML course "Machine Learning for Graduate Students":
- **Course**: `Main_Course_MachineLearning_{Presentation,CheatSheet}.tex` → `course_machinelearning_content.tex`
- **Workshops** (6 × driver pairs):
  - W1 Python for ML (8h): existing `workshop_python_basic_content.tex` (renamed from `workshop_python_content.tex` in Aug 2026, see Python course note below)
  - W2 Foundations (4h): `workshop_ml_foundations_content.tex`
  - W3 Regression (4h): `workshop_ml_regression_content.tex`
  - W4 Tree-Based & Ensemble (8h): `workshop_ml_treebased_content.tex`
  - W5 Supervised II — KNN/SVM/NB (8h): `workshop_ml_supervisedII_content.tex`
  - W6 Unsupervised & Deployment (8h): `workshop_ml_unsupervised_content.tex`
  - Standalone all-ML workshop (W2–W6, no Python/demos): `Main_Workshop_MachineLearning_{Presentation,CheatSheet}.tex`
- **Seminars** (10 × driver pairs): `seminar_ml_{intro,dataprep,regression,decisiontree,ensemble,knn,svm_nb,clustering,dimreduction,deployment}_content.tex`; drivers are `Main_Seminar_ML_{Intro,DataPrep,Regression,DecisionTree,Ensemble,KNN,SVM_NB,Clustering,DimReduction,Deployment}_{Presentation,CheatSheet}.tex`
- **New demo/assign files**: `ml_course_demo_regression_housing.tex`, `ml_course_demo_svm_digits.tex`, `ml_course_assign_knn_wine.tex`, `ml_course_demo_clustering_customers.tex`, `ml_course_assign_pca_digits.tex`
- **Upgrade status**: Seminar 1 (Intro) upgraded; track remaining 9 in `LaTeX/todo_ml_seminar_upgrade.md`
- **Pending**: `course_machinelearning_content.tex` still needs the 5 new demo/assign files listed above added at the end

### Maths for ML restructured (July 2026), promoted to a course (Aug 2026)
Full 4-level hierarchy for "Zero-to-Hero: Mathematics for Machine Learning", aimed at
fresher/college-level students. 12 seminars × ~2h = 24h ≈ 3 days × 8h, but the 4 topic-workshops
are uneven in size (Basics/LinearAlgebra/Calculus 4h each, Statistics 12h), so the day boundary
cuts across the Calculus/Statistics workshop pair rather than aligning 1:1 with workshops —
annotated as comments in `course_maths4ml_content.tex` and inside
`workshop_maths4ml_statistics_content.tex` (not a structural split):
- **Day 1 (8h)**: Basics + Linear Algebra
- **Day 2 (8h)**: Calculus + Statistics seminars 1–2 (probability_foundations, random_distributions)
- **Day 3 (8h)**: Statistics seminars 3–6 (centraltendency_spread, distributions_expectedvalue, hypothesis_testing, tests_practice)
- **Course**: `Main_Course_MathsML_{Presentation,CheatSheet}.tex` → `course_maths4ml_content.tex`
- **Workshops** (4 × driver pairs), each just chaining its seminars:
  - Basics: `workshop_maths4ml_basics_content.tex` → `seminar_maths4ml_basics_{numbers_equations,sets_proofs}_content.tex`; drivers `Main_Workshop_MathsML_Basics_{Presentation,CheatSheet}.tex`
  - Linear Algebra: `workshop_maths4ml_linearalgebra_content.tex` → `seminar_maths4ml_linearalgebra_{vectors,matrices}_content.tex`; drivers `Main_Workshop_MathsML_LinearAlgebra_{Presentation,CheatSheet}.tex`
  - Calculus: `workshop_maths4ml_calculus_content.tex` → `seminar_maths4ml_calculus_{functions_limits,derivatives_optimization}_content.tex`; drivers `Main_Workshop_MathsML_Calculus_{Presentation,CheatSheet}.tex`
  - Statistics (6 seminars): `workshop_maths4ml_statistics_content.tex` → `seminar_maths4ml_statistics_{probability_foundations,random_distributions,centraltendency_spread,distributions_expectedvalue,hypothesis_testing,tests_practice}_content.tex`; drivers `Main_Workshop_MathsML_Statistics_{Presentation,CheatSheet}.tex`
- **Seminars** (12 × driver pairs, unchanged): each has its own driver pair
  `Main_Seminar_MathsML_<ParentTopic>_<Subtopic>_{Presentation,CheatSheet}.tex`
- All 12 seminars have been through an intuition-first `/upgrade-deck` pass (technical fixes,
  "Intuition" callouts, section-end "Quick Check" quizzes) — see git history for details, as
  `LaTeX/todo_maths4ml_seminar_upgrade.md` (the working to-do for this restructuring) was
  deleted once the work completed.
- Raw `maths_*.tex` topic files are unchanged; only the aggregation layers changed.
- The old single all-in-one `Main_Workshop_ML_Maths_{Presentation,Cheatsheet}.tex` /
  `workshop_maths4ml_content.tex` were removed as redundant once the course/workshop split
  landed (unlike the ML course, no standalone "complete workshop" was kept here).

### Python course added (Aug 2026), seminar layer added
2-day, 16h course combining the two existing standalone Python workshops as Day 1 / Day 2:
- **Course**: `Main_Course_Python_{Presentation,CheatSheet}.tex` → `course_python_content.tex`
- **Day 1 (8h)**: `workshop_python_basic_content.tex` (renamed from `workshop_python_content.tex`;
  also still used standalone via `Main_Workshop_Python_Basic_{Presentation,CheatSheet}.tex`, and
  as W1 "Python for ML" in `course_machinelearning_content.tex`)
- **Day 2 (8h)**: `workshop_python_adv_content.tex` (unchanged; also still used standalone via
  `Main_Workshop_Python_Advanced_{Presentation,CheatSheet}.tex`)
- Both workshops now route through a seminar layer (6 seminars each) between the workshop and
  the raw `python_*.tex` topic files, matching the Maths4ML/ML hierarchy:
  - **Basic** (`workshop_python_basic_content.tex`): B1 Intro, B2 Constructs, B3 Procedures,
    B4 OOP, B5 IOLibraries, B6 Closure — `seminar_python_basic_<name>_content.tex`; drivers
    `Main_Seminar_Python_Basic_<Name>_{Presentation,CheatSheet}.tex`
  - **Advanced** (`workshop_python_adv_content.tex`): A1 OOPIteration, A2 FunctionsOS,
    A3 StringsWeb, A4 DataLibs, A5 Visualization, A6 Problems —
    `seminar_python_adv_<name>_content.tex`; drivers
    `Main_Seminar_Python_Advanced_<Name>_{Presentation,CheatSheet}.tex`
  - `Extra`/`References` sections stay as trailing raw `\input`s in the workshop content files,
    not wrapped in a seminar (same precedent as Maths4ML/ML)
  - `python_oop.tex` is `\input` by both Basic B4 and Advanced A1 — duplication predates this
    restructure, preserved as-is
- Raw `python_*.tex` topic files are unchanged; only the aggregation layers changed.
- **Upgrade status**: Basic B1 (Intro), B2 (Constructs), B3 (Procedures) have been through an
  intuition-first `/upgrade-deck` pass (technical fixes, "Intuition" callouts, "Quick Check"
  quizzes); B4-B6 and Advanced A1-A6 remain — track progress in
  `LaTeX/todo_python_seminar_restructure.md`
- No redundant files removed here: unlike Maths4ML, both standalone workshops remain valid
  independent offerings, so nothing was retired.

### CareerInDataScience seminar split into 90-min full + 30-min short (Oct 2026)
First precedent in the repo for a single seminar offered at two durations, sharing
underlying topic files kept in sync by commenting rather than by duplicating content
independently:
- **Full (90 min, unchanged)**: `Main_Seminar_Tech_CareerInDataScience_{Presentation,CheatSheet}.tex`
  → `seminar_careerindatascience_content.tex` (Background, Introduction, Challenges,
  Roles \& Personas, Preparation, Mid-career, References)
- **Short (30 min, new)**: `Main_Seminar_Tech_CareerInDataScience_Short_{Presentation,CheatSheet}.tex`
  → `seminar_careerindatascience_short_content.tex`, dropping Background/Challenges/
  Mid-career entirely (their `\section`+`\input` lines commented out, not deleted) and
  swapping the rest to `_short` topic files: `ai_intro_tech_short.tex` (~11 orientation
  slides), `career_ai_roles_short.tex` (8 of ~16 roles), `career_ai_personas_short.tex`
  (all 3 personas, trimmed detail), `career_ai_prep_short.tex` (5 of ~12 slides);
  `career_refs.tex` kept in full (only 2 frames)
- **Sync convention**: each `_short.tex` sibling is a full copy of its parent with the
  excluded frames commented out (not rewritten/deleted), so a frame added to the parent
  can be manually mirrored into the child as either live or commented — the same
  discipline as the pre-existing repo-wide `X.tex`/`X_short.tex` pattern (e.g.
  `dl_intro.tex`/`dl_intro_short.tex`), now extended to a full seminar-level split
- `ai_intro_tech_short.tex` — shared by 5 other decks (`course_deeplearning_content`,
  `seminar_artificialintelligencemachinelearning_content`,
  `seminar_artificialintelligence_tech_content`, `seminar_machinelearning_content`, and
  commented in `seminar_llm_genai_content`) — was renamed to `ai_intro_tech.tex` first
  (content-preserving; it was never actually short, just misnamed) so a genuine
  ~11-slide `ai_intro_tech_short.tex` could be created without touching those decks
- Both variants went through an `/upgrade-deck` pass, each reviewed as its own
  standalone artifact; working notes (now complete) were in
  `LaTeX/todo_careerindatascience_split.md`

### Short/full topic-file sync audit (Oct 2026)
Repo-wide audit of all `X.tex`/`X_short.tex` pairs against the comment-sibling
convention (see CareerInDataScience note above). Of 24 pairs checked, 13 were
already in sync, 3 near-miss pairs and 3 missing-only pairs got small content
fixes, and 5 pairs had drifted so far apart (independently authored, not a
subset relationship at all) that they were split into a three-file group —
**this introduces a new file-naming pattern**: `<topic>_overview.tex` now
denotes a deep/comprehensive standalone treatment that sits *outside* the
comment-sibling relationship, while `<topic>.tex`/`<topic>_short.tex` were
freshly authored as a genuine (smaller) comment-sibling pair distilled from
it. The five:
- `dnlp_intro_overview.tex` (renamed from the old `dnlp_intro.tex`) —
  Kirill Eremenko technical walkthrough (Seq2Seq/Attention/Decoding), used by
  `workshop_deepnlp_content.tex`. Fresh `dnlp_intro.tex`/`_short.tex` merge in
  the old short's foundational material (Turing Test, NLP tasks, embeddings
  basics), used by `seminar_deepnaturallanguageprocessing_content.tex`.
- `data_intro_overview.tex` (renamed from the old `data_intro.tex`) —
  technical data-types/distance-metrics deep dive (NOIR, Euclidean/Minkowski,
  SMC, Cosine Similarity), used by `seminar_data_tensorflow_content.tex`'s
  "Basic Concepts in Data" section (a good label fit, unlike before). Fresh
  `data_intro.tex`/`_short.tex` are a motivational/historical intro (Man on
  the Moon, "Data is the New Oil", the 4 Vs, Target case study), used by
  `workshop_dataanalytics_content.tex`'s "Introduction" section (kept on the
  plain filename — a deliberate deviation from the mechanical rename, since
  the old full's technical content never fit that section's actual title).
- `dl_intro_overview.tex` (renamed from the old `dl_intro.tex`) — the deeper
  synthesis (still includes backprop math, automatic differentiation,
  optimizer plots), used by `seminar_deeplearning_foundations_content.tex`.
  Fresh `dl_intro.tex`/`_short.tex` condense both old files' strengths into a
  non-mathematical walkthrough, used by `seminar_deeplearning_content.tex`.
- `python_syntax_overview.tex` (renamed from the old `python_syntax.tex`,
  content unchanged) — this one had already been through `/upgrade-deck` as
  part of the Python seminar restructuring's B1 (`seminar_python_basic_intro_
  content.tex`); the rename was verified to preserve that work exactly (same
  118-page compile). Fresh `python_syntax.tex`/`_short.tex` are a broader
  "Python Basics" primer (the old short's actual scope, despite its narrow
  name), used by `course_deeplearning_content.tex`'s W1 recap.
- `nlp_embedding_overview.tex` (renamed from the old `nlp_embedding.tex`) —
  the authoritative, current (2023-2026) material: BERT, RAG systems,
  multimodal/CLIP, bias/ethics, a full Tweet-Sentiment-with-Word2Vec code case
  study. Used by `seminar_wordembeddings_content.tex` (a seminar wholly about
  embeddings, so its "Introduction" section is really the whole seminar's
  substance — repointed here rather than mechanically). Fresh
  `nlp_embedding.tex`/`_short.tex` condense the overview's structure, used by
  `seminar_nlp_advanced_content.tex` (one topic among several in a broader
  workshop) and `workshop_deepnlp_content.tex`.
- Working notes (now complete) were in `LaTeX/todo_short_full_sync_audit.md`,
  deleted once the work finished — same precedent as the other `todo_*`
  restructuring files noted elsewhere in this document.

**Dedicated standalone seminars for 4 of the 5 overview files (Oct 2026,
additive follow-up)**: each of the 5 `_overview.tex` files is used by an
existing consumer deck that specifically needs its depth (documented above)
— those consumers were deliberately left as-is, not repointed. Instead, 4 of
the 5 also got a *new*, minimal, single-section standalone seminar so the
deep-dive content is independently reachable as its own session (the 5th,
`nlp_embedding_overview.tex`, already had one — `seminar_wordembeddings_
content.tex`'s only real section is the overview, backed by `Main_Seminar_
NLP_WordEmbeddings_{Presentation,CheatSheet}.tex`):
- `dnlp_intro_overview.tex` → `seminar_dnlp_overview_content.tex` →
  `Main_Seminar_NLP_DNLP_Overview_{Presentation,CheatSheet}.tex`
- `data_intro_overview.tex` → `seminar_dataconcepts_overview_content.tex` →
  `Main_Seminar_Data_Concepts_Overview_{Presentation,CheatSheet}.tex`
- `dl_intro_overview.tex` → `seminar_dl_technical_overview_content.tex` →
  `Main_Seminar_DL_Foundations_Overview_{Presentation,CheatSheet}.tex`
- `python_syntax_overview.tex` → `seminar_python_syntax_overview_content.tex`
  → `Main_Seminar_Python_Syntax_Overview_{Presentation,CheatSheet}.tex`

Each content wrapper is deliberately minimal: a single `\section[Overview]
{Overview}` + `\input{<topic>_overview}`, no References section (unlike most
seminars, by design — these are supplementary deep-dive sessions, not full
independent courses). All 8 driver files compiled clean.

### Adding a new topic
1. Create `LaTeX/<domain>_<topic>.tex` with Beamer frames
2. `\input{<domain>_<topic>}` inside the relevant `seminar_*_content.tex`
3. Place supporting images in `LaTeX/images/` (5000+ images already there, mostly PDFs)

### Frame boilerplate
```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{frame}[fragile]\frametitle{Slide Title}
\begin{itemize}
\item Point one
\end{itemize}
\end{frame}
```

For section dividers:
```latex
\begin{frame}[fragile]\frametitle{}
\begin{center}
{\Large Section Heading}
\end{center}
\end{frame}
```

## Code Directory

Each subdirectory under `Code/` corresponds to a library or topic. No single build or test command — run scripts individually per subdirectory.

### Environment setup (conda-based)
Every major subdirectory has an `environment.yml`. The standard setup flow is:
```bash
conda env create -f Code/<subdir>/environment.yml
conda activate <env-name>
```
Do not create `venv/` or `.venv/` folders — use conda environments only.

### Key subdirectory map

| Category | Directories |
|----------|-------------|
| GenAI / Agents | `langchain/`, `langgraph/`, `llamaindex/`, `crewai/`, `agents/`, `agno/`, `google-adk/` |
| RAG Applications | `chatbot-faqs/`, `chatbot-multimodal/`, `omni-rag/`, `parsing/`, `graphrag/` |
| LLM Fine-tuning | `fine-tuning/`, `ludwig/`, `gemma/` |
| Document Parsing | `docling/`, `opendataloader/` |
| Deep Learning | `keras/`, `dl_tf2/`, `pytorch/`, `deep_rl/` |
| Classical ML | `ml/`, `math/`, `python/` |
| NLP | `nlp/`, `dnlp/`, `spacy/` |
| GNN | `pyg/` |
| Indic Language | `mahamarathi/`, `sarvam/`, `orgpedia/` |
| Research Refs | `txt2cad/`, `txt2sql/` (docs only, no runnable code) |

### Code/.gitignore
A repo-wide `Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, model weights (`*.bin`, `*.pt`, `*.safetensors`).

### Notable sub-projects with their own config
- `Code/claudecode/MyWorkshop/` and `Code/claudecode/CadCamWorkshop/` — each has its own `CLAUDE.md` (no `CLAUDE.md` directly under `Code/claudecode/` itself)
- `Code/langgraph/open_deep_research-langcahin-ai/` — has its own `CLAUDE.md` and `README.md`
- `Code/crewai/researcher/` — uses `pyproject.toml` + `uv.lock` (modern uv workflow)

### Security note
`Code/google-adk/my_agent/.env` is gitignored but contains a real `GOOGLE_API_KEY` on disk — rotate it in Google Cloud Console.

## Test Suite

All Python-script directories have a `test_*.py` file runnable with `pytest` in the `genai` conda environment.

### Running tests

Run a single suite:
```bash
conda activate genai
cd Code/<subdir>
python -m pytest test_*.py -v
```

Run all suites together (from repo root):
```bash
conda run -n genai python -m pytest \
  Code/graphrag/test_graphrag.py \
  Code/parsing/test_parsing.py \
  Code/agno/test_agno.py \
  Code/google-adk/test_tools.py \
  Code/chatbot-faqs/test_chatbot_faqs.py \
  Code/chatbot-multimodal/test_models.py \
  Code/omni-rag/test_omnirag.py \
  -v
```

### Test files per directory

| Directory | Test file | Tests | What's covered |
|-----------|-----------|-------|----------------|
| `chatbot-faqs/` | `test_chatbot_faqs.py` | 14 | CSV loading, similarity threshold, cosine similarity logic |
| `chatbot-multimodal/` | `test_models.py` | 19 | Pydantic chunk models, DoclingParser device selection, null-safe heading join |
| `omni-rag/` | `test_omnirag.py` | 9 | Context list-join fix, OmniIngestor structure (mocked), ragas/datasets imports |
| `parsing/` | `test_parsing.py` | 12 | GroqResumeParser: empty-key validation, default model, mock API call |
| `graphrag/` | `test_graphrag.py` | 9 | `distance()` boundary conditions, networkx/pandas integration |
| `google-adk/` | `test_tools.py` | 10 | Tool functions (web_search, get_stock_price, etc.) with mocked yfinance |
| `agno/` | `test_agno.py` | 7 | agno package imports, syntax validation of all .py files |

### Test design notes
- No real API calls — all LLM/embedding clients are mocked with `unittest.mock`.
- No model downloads — `transformers` model-loading calls are patched at the function level.
- The `google-adk` tests mock the `adk` package (not installed on all machines).
- The omni-rag `TestOmniIngestorStructure` tests skip gracefully if a `datasets` circular import occurs when running in a combined pytest session (they pass in isolation).
- `ragas` and `google-adk` packages were added to the `genai` env during the April 2026 upgrade pass.

### Known environment notes
- `ragas 0.4.3` upgraded `openai` from 1.x → 2.x — verify `langchain-openai` compatibility if issues arise.
- A broken system-Python `faiss` install exists at `C:\Users\yoges\AppData\Roaming\Python\Python310\site-packages\faiss\` and conflicts if imported outside the conda env.
- `opendataloader-pdf` and `langchain-opendataloader-pdf` are installed in the `genai` env (added May 2026). The library wraps a Java JAR — **Java 11+ must be on PATH** before any tutorial runs. Install via `conda install -n genai -c conda-forge openjdk=11`. Tutorial 09 (OCR) additionally requires the hybrid backend started in a separate terminal: `opendataloader-pdf-hybrid --port 5002 --force-ocr`.

## Memory
Do not store, write, or update any memory files in the global `~/.claude/projects/` directory unless the user explicitly confirms or allows it in the current conversation.

## Git
Do not run any git commands. The user manages all git operations externally.
