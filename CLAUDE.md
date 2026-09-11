# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

An open-source educational repository containing:
- **LaTeX**: Beamer presentation slides and two-column cheatsheets for Data Science courses (Python, ML, DL, NLP, GenAI, RAG, etc.)
- **Code**: Python scripts and Jupyter notebooks demonstrating the concepts covered in the slides

## Style Conventions

User-visible markdown (`README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `COURSES.md`,
`Code/README.md`, and similar reader-facing docs) must not use em-dashes; use a colon,
comma, or semicolon instead. Internal/tooling files (this `CLAUDE.md`, other `CLAUDE.md`s,
templates) are exempt.

**Automated enforcement**: a `PreToolUse` hook (`.claude/hooks/check-em-dash.ps1`, wired in
`.claude/settings.json`) blocks any `Edit`/`Write` to a `.tex` or `.md` file (any filename
except `CLAUDE.md` itself) whose new content contains a literal em-dash, scanning outside
`%`-comments and `lstlisting`/`verbatim` blocks. Scope is broader than the paragraph above:
it covers every `.tex` file repo-wide, not just user-visible markdown. A literal `--`/`---`
outside those exclusions triggers a non-blocking warning instead. The hook reads stdin via a
UTF-8-forced `StreamReader`, not `[Console]::In` (the latter mangles the em-dash's multi-byte
UTF-8 encoding through the console's legacy codepage). If the hook stops catching em-dashes,
suspect this class of bug first.

## LaTeX Build System

Compile a specific deck from the `LaTeX/` directory using MikTeX's `texify`:
```bat
cd LaTeX
texify -cp Main_Seminar_AI_HandsOn_ClaudeCode_Presentation.tex
```
Compile all decks matching a pattern (Windows):
```bat
cd LaTeX
for /r %i in (Main_Seminar_*Educators*.tex) do texify -cp %i
```
Compile everything: `LaTeX/make_all.bat`

### Compiled output policy
This repo is public; **compiled decks (`Main_*.pdf`) and build litter (`.aux .log .nav .out
.snm .toc .vrb`) are never committed here** (gitignored at `LaTeX/.gitignore`). Final PDFs for
delivery get copied to the private `Publications/Presentations/` repo after compiling locally.
`LaTeX/images/*.pdf` are source assets, not build output, and are unaffected by this rule.

## LaTeX Architecture

### 4-level content hierarchy
```
Course (40hr)     Main_Course_*_{Presentation,CheatSheet}.tex
                    -> course_*_content.tex
                         -> \input{workshop_*_content}  (+ course-specific extras)

Workshop (4-16hr) Main_Workshop_*_{Presentation,CheatSheet}.tex
                    -> workshop_*_content.tex
                         -> \input{seminar_*_content}  (seminar layer only)

Seminar (1hr)     Main_Seminar_*_{Presentation,CheatSheet}.tex
                    -> seminar_*_content.tex
                         -> \input{<domain>_<topic>}  (raw topic files)
```
Every deliverable has two output forms sharing the same content file:
- `Main_*_Presentation.tex` -- Beamer slides (`\documentclass{beamer}`, `template_presentation.tex`)
- `Main_*_CheatSheet.tex` -- Two-column landscape notes (`\documentclass{article}`, `template_cheatsheet.tex`)

CheatSheet column count: Seminars use `multicols{3}`; Workshops use `multicols{2}`.

**Never use a float (`table[h]`, `figure`) in any content file that feeds a CheatSheet.**
Floats can't be placed inside `multicols` and are silently *dropped* -- no error, no warning,
the content just vanishes from that PDF while the Presentation renders fine. Use
`\begin{center}` + a bare `tabular` instead. Only caught by rendering the PDF to an image, not
by the compile log.

**Two-column `adjustbox`+`minipage` frames need `%` line-endings.** The repo's side-by-side
convention (explanatory content left ~0.55`\linewidth`, diagram right ~0.4`\linewidth`) sums to
~0.96`\linewidth`, leaving no room for LaTeX's inter-line spaces at each newline. Terminate
those lines with `%` (`\end{minipage}%`, `}%`, `\hfill%`, `\adjustbox{valign=t}{%`), or the
right minipage silently wraps *below* the left one and runs off the bottom of the slide -- the
only log signal is a single `Overfull \vbox`, easy to miss. This has recurred repeatedly even
after being documented -- apply it preemptively on every new two-column frame, and always
render the page to confirm rather than trusting that the fix was applied correctly.

`\usepackage{beamerarticle}` in `template_cheatsheet.tex` makes Beamer frames compile in
article mode with no frame stripping needed. Both templates load `\usepackage{upquote}` right
after `listings` so code-block quotes/backticks render straight, not curly.

**Silent vertical overflow is common and never warns in the log.** A TikZ diagram or a code
block followed immediately by prose can overflow a frame, or a narrow 3-column CheatSheet
column, with zero `Overfull`/`Underfull` warning. Always render suspect pages to an image
(`pdftoppm -png -r 100 -f N -l N`) and look, especially after adding any diagram or code+text
combination. For diagrams under the two-column convention, wrap in
`\adjustbox{max width=\linewidth, max totalheight=<value>}` from the start -- width-only
capping does nothing for vertical overflow.

`template_presentation.tex`'s `lstset` carries `belowskip=-15pt`: any body text placed directly
after `\end{lstlisting}` gets pulled up into the code box, often with no warning either. Fold
a short trailing sentence into the listing as a final `#` comment line rather than fighting
`\vspace`.

### Naming conventions
- Topic files: `<domain>_<topic>.tex` (e.g., `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Driver files: `Main_[Course|Seminar|Workshop]_<Subject>_[Presentation|CheatSheet].tex`
  (Seminar ~= 1 hour, Workshop ~= 1 day, Course ~= 1 week/semester)
- Every Seminar and Workshop **must** have both a `_Presentation.tex` and `_CheatSheet.tex` driver
- `_Short` suffix on a **driver** denotes a shorter-duration variant of an existing seminar,
  sharing topic files with its parent via the `X.tex`/`X_short.tex` comment-sync pattern below
- `_overview` suffix on a **topic file** denotes a deep/comprehensive standalone treatment that
  sits outside the `X.tex`/`X_short.tex` comment-sibling relationship
- `_Overview` suffix on a **driver** denotes a minimal, single-section standalone seminar that
  makes an `_overview.tex` topic file independently reachable as its own session (no References
  section, by design)

### Sibling-file sync rule (standing rule)
Whenever a `.tex` file being edited has an `X.tex`/`X_short.tex` comment-sibling, check for
that sibling and read it too, even if the driver you were pointed at doesn't `\input` it. Any
frame added/removed/materially edited in one must have its comment/uncomment state mirrored in
the other, so they never silently drift. Built into `/upgrade-deck` (Step 4), but applies to
any manual edit too. **Always count *live* (uncommented) frames, not raw `\begin{frame}`
occurrences, when judging a deck's size** -- many files have large chunks commented out by the
original author (grep counts have been off by 40%+).

**User preference, confirmed repo-wide**: default to the full (`X.tex`) sibling over `_short`
wherever there's no genuine duration-based reason for the short variant -- several `_short`
files were found receiving all the upgrade work while every live deck still pointed at a
stale full sibling (or vice versa); always check both before assuming one is current.

**Always invoke `/upgrade-deck` itself rather than reconstructing its Task 1-6 checklist from
memory.** A manual pass following the same steps from recollection has silently skipped
checklist items (e.g. the prose-dash sweep) with no error signal, only caught later by the user
spotting the miss visually.

### Known issues
- **Accepted, deprioritized: git-index/disk casing drift on the 8 `Main_Seminar_AI_For_*`
  seminar drivers** (WithML, Kids, BizLeaders, ProjectManagers, TechLeaders, Educators, All_Tech,
  All_NonTech). Confirmed for Educators: files on disk are `AI_For_Educators` (capital F,
  matching `COURSES.md`), git's tracked index path is `AI_for_Educators` (lowercase) -- a
  case-only rename invisible to `git status` on a case-insensitive Windows checkout. Harmless on
  Windows; would only break on a case-sensitive clone (Linux, CI). A `git mv` fix was attempted
  and abandoned. Revisit only if a case-sensitive checkout is ever actually needed.
- `Main_Course_GenerativeAI_{Presentation,CheatSheet}` has never been successfully compiled end
  to end (its `\input` chain resolves, but a full compile attempt was cancelled after 10+
  minutes with a 35MB+ PDF still growing). Do not record it as compiling until someone actually
  builds it to completion.
- Repo-wide `\input`-resolution is otherwise clean: a static walk of every driver's `\input`
  chain reports 0 unresolved targets across all current drivers. Worth re-running
  (`latex-audit inputs`) after any bulk rename.

### Frame boilerplate
```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{frame}[fragile]\frametitle{Slide Title}
\begin{itemize}
\item Point one
\end{itemize}
\end{frame}
```
Section dividers:
```latex
\begin{frame}[fragile]\frametitle{}
\begin{center}
{\Large Section Heading}
\end{center}
\end{frame}
```

### Adding a new topic
1. Create `LaTeX/<domain>_<topic>.tex` with Beamer frames
2. `\input{<domain>_<topic>}` inside the relevant `seminar_*_content.tex`
3. Place supporting images in `LaTeX/images/` (5000+ images already there, mostly PDFs)

## Current course structure (reference)

- **Machine Learning** (40hr course): `course_machinelearning_content.tex` -> 6 workshops (Python
  for ML, Foundations, Regression, Tree-Based & Ensemble, Supervised II, Unsupervised &
  Deployment) -> 10 seminars (`seminar_ml_{intro,dataprep,regression,decisiontree,ensemble,
  knn,svm_nb,clustering,dimreduction,deployment}_content.tex`).
- **Maths for ML** (24hr course): `course_maths4ml_content.tex` -> 4 workshops (Basics, Linear
  Algebra, Calculus, Statistics) -> 12 seminars, each `Main_Seminar_MathsML_<Topic>_<Subtopic>_*`.
- **Python** (16hr course, 2 days): `course_python_content.tex` -> Basic (6 seminars, B1-B6) and
  Advanced (6 seminars, A1-A6) workshops, also usable standalone.
- **Reinforcement Learning**: `workshop_reinforcementlearning_content.tex` -> 7 seminars
  (Introduction, Concepts, MDP, Q-Learning, Deep Q-Learning & Modern RL, Tools & Frameworks,
  Applications), each `Main_Seminar_ML_ReinforcementLearning_<SubTopic>_*`.
- **CareerInDataScience**: offered at two durations sharing topic files via the comment-sync
  pattern -- Full (90 min, `seminar_careerindatascience_content.tex`) and Short (30 min,
  `seminar_careerindatascience_short_content.tex`).
- **MLCoEP** ("AI-ML for Mechanical Engineers", bespoke CoEP course): 20 independent sessions,
  each `seminar_mlcoep_session_<N>_content.tex` + its own driver pair
  `Main_Seminar_MLCoEP_Session_<N>_<ShortName>_{Presentation,CheatSheet}.tex`. All 20 have been
  through a full `/upgrade-deck` pass. Per-session drivers physically live in
  `LaTeX/_retired/mlcoep_session_drivers/` but remain the **active verification workflow**:
  copy the pair to `LaTeX/` root, compile, verify, copy the PDF to `Publications/Presentations/`,
  then re-retire -- see `project_mlcoep_session_driver_workflow` in assistant memory.
  `course_mlcoep_content.tex` chains all 20 session files into one combined `Main_Course_MLCoEP_*`
  driver; **never compile that combined driver without asking first** (it's a 20+ minute,
  350+ page job, and its own correctness has never been verified).
- `_overview.tex` topic files (deep-dive standalone treatments, outside the short/full sibling
  pattern): `dnlp_intro_overview`, `data_intro_overview`, `dl_intro_overview`,
  `nlp_embedding_overview` -- each also has its own minimal standalone `_Overview` seminar
  driver so it's independently reachable.

### Local LLM (Qwen3) as a Groq alternative -- piloted pattern
`ChatLlamaCpp` (langchain_community, in-process, no server) with a local
`Qwen3-1.7B-Q4_K_M.gguf` works as a drop-in for `ChatGroq` in plain multi-turn chat (no tool
calling) -- validated pattern lives in `Code/langchain/langchain_v1_models.py` and
`Code/omni-rag/agent.py`. Requires `llama-cpp-python>=0.3.34` (older versions don't know the
Qwen3 architecture) and appending `/no_think` to the prompt (Qwen3 otherwise burns the whole
token budget on hidden "thinking" and returns truncated garbage). **`bind_tools()` with
`tool_choice="auto"` does not reliably work at this model size** -- the 1.7B model reasons about
tools in its thinking block but doesn't trigger auto tool-detection; a forced `tool_choice`
naming a specific function does work. `Code/langchain/langchain_v1_createagent.py` (the one
genuinely tool-calling site) stays on `ChatGroq` for this reason. Not rolled out to the other
`ChatGroq` call sites in `Code/` -- this remains an opt-in pattern, not a default.

## Code Directory

Each subdirectory under `Code/` corresponds to a library or topic. No single build or test command -- run scripts individually per subdirectory.

### Environment setup (conda-based)
Every major subdirectory has an `environment.yml`. Standard flow:
```bash
conda env create -f Code/<subdir>/environment.yml
conda activate <env-name>
```
Do not create `venv/` or `.venv/` folders -- use conda environments only.

### Key subdirectory map

| Category | Directories |
|----------|-------------|
| GenAI / Agents | `langchain/`, `langgraph/`, `llamaindex/`, `crewai/`, `agents/`, `agno/`, `google-adk/` |
| RAG Applications | `chatbot-faqs/`, `chatbot-multimodal/`, `omni-rag/`, `parsing/`, `graphrag/` |
| LLM Fine-tuning | `fine-tuning/` |
| Document Parsing | `docling/`, `opendataloader/` |
| Deep Learning | `pytorch/` |
| Classical ML | `ml/`, `math/`, `python/` |
| NLP | `nlp/`, `dnlp/`, `spacy/` |
| GNN | `gnn/pyg/`, `gnn/gnn-project-deepfindr/`, `gnn/molecule-deepfindr/`, `gnn/odsc2021-sujitpal/` |
| Indic Language | `mahamarathi/`, `sarvam/`, `orgpedia/` |
| Research Refs | `txt2cad/`, `txt2sql/` (docs only, no runnable code) |
| Other | `amd/` (AMD Academy course materials), `chromeext/` (small Chrome extension side-project) |
| Reference-only (not this repo) | `keras/`, `dl_tf2/`, `Admin/diagram_sources/` moved to a sibling `TeachingDataScience_removed_thirdparty/` folder -- third-party tutorial/book content with no proper attribution |

`Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, model weights
(`*.bin`, `*.pt`, `*.safetensors`), plus `node_modules/`, `.pytest_cache/`, `.ruff_cache/`,
`.benchmarks/`.

### Notable sub-projects with their own config
- `Code/claudecode/MyWorkshop/` and `Code/claudecode/CadCamWorkshop/` -- each has its own `CLAUDE.md`
- `Code/langgraph/open_deep_research-langcahin-ai/` -- has its own `CLAUDE.md` and `README.md`
- `Code/crewai/researcher/` -- uses `pyproject.toml` + `uv.lock` (uv workflow)

### Security note
`Code/google-adk/my_agent/.env` is gitignored but contains a real `GOOGLE_API_KEY` on disk -- rotate it in Google Cloud Console.

## Test Suite

All Python-script directories have a `test_*.py` file runnable with `pytest` in the `genai` conda environment.

### Running tests
Single suite:
```bash
conda activate genai
cd Code/<subdir>
python -m pytest test_*.py -v
```
All suites together (from repo root):
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
- No real API calls -- all LLM/embedding clients are mocked with `unittest.mock`.
- No model downloads -- `transformers` model-loading calls are patched at the function level.
- The `google-adk` tests mock the `adk` package (not installed on all machines).
- The omni-rag `TestOmniIngestorStructure` tests skip gracefully if a `datasets` circular import occurs in a combined pytest session (they pass in isolation).
- `ragas` and `google-adk` packages are in the `genai` env.

### Known environment notes
- `ragas 0.4.3` upgraded `openai` from 1.x -> 2.x -- verify `langchain-openai` compatibility if issues arise.
- A broken system-Python `faiss` install exists at `C:\Users\yoges\AppData\Roaming\Python\Python310\site-packages\faiss\` and conflicts if imported outside the conda env.
- `opendataloader-pdf` and `langchain-opendataloader-pdf` are in the `genai` env; the library wraps a Java JAR -- **Java 11+ must be on PATH** before any tutorial runs (`conda install -n genai -c conda-forge openjdk=11`). Tutorial 09 (OCR) additionally requires the hybrid backend: `opendataloader-pdf-hybrid --port 5002 --force-ocr`.

## Memory
Do not store, write, or update any memory files in the global `~/.claude/projects/` directory unless the user explicitly confirms or allows it in the current conversation.

## Git
Do not run any git commands. The user manages all git operations externally.
