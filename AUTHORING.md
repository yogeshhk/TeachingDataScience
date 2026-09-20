# Authoring and development notes (TeachingDataScience)

Detailed conventions for the LaTeX decks and the `Code/` projects. `CLAUDE.md` at the repo root holds the
short rules and points here; read this file when authoring, compiling or testing.

## Repository Purpose

- **LaTeX**: Beamer slides and two-column cheatsheets for Data Science courses (Python, ML, DL,
  NLP, GenAI, RAG, etc.)
- **Code**: Python scripts and notebooks demonstrating the concepts in the slides

## Style Conventions

User-visible markdown (`README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `COURSES.md`,
`Code/README.md`, and similar) must not use em-dashes; use a colon, comma or semicolon. Internal
files (`CLAUDE.md`, templates) are exempt.

A `PreToolUse` hook (`.claude/hooks/check-em-dash.ps1`, wired in `.claude/settings.json`) blocks
any `Edit`/`Write` to a `.tex` or `.md` file (except `CLAUDE.md`) whose new content contains a
literal em-dash outside `%` comments and `lstlisting`/`verbatim` blocks. A literal `--`/`---`
outside those triggers a non-blocking warning. The hook reads stdin through a UTF-8 `StreamReader`,
not `[Console]::In`, which mangles multi-byte UTF-8 through the legacy codepage. If it stops
catching em-dashes, suspect that first.

## LaTeX Build System

Compile from the `LaTeX/` directory with MikTeX's `texify`:
```bat
cd LaTeX
texify -cp Main_Seminar_AI_HandsOn_ClaudeCode_Presentation.tex
for /r %i in (Main_Seminar_*Educators*.tex) do texify -cp %i
make_all.bat
```

### Compiled output policy
This repo is public. **Compiled decks (`Main_*.pdf`) and build litter (`.aux .log .nav .out .snm
.toc .vrb`) are never committed** (gitignored at `LaTeX/.gitignore`). Copy final PDFs to
`Publications/Presentations/` after compiling locally. `LaTeX/images/*.pdf` are source assets and
are unaffected.

## LaTeX Architecture

### 3-level content hierarchy
```
Course (40hr)     Main_Course_*_{Presentation,CheatSheet}.tex
                    -> course_*_content.tex -> \input{workshop_*_content}
Workshop (4-16hr) Main_Workshop_*_{Presentation,CheatSheet}.tex
                    -> workshop_*_content.tex -> \input{seminar_*_content}
Seminar (1hr)     Main_Seminar_*_{Presentation,CheatSheet}.tex
                    -> seminar_*_content.tex -> \input{<domain>_<topic>}
```
Every deliverable has two outputs sharing one content file:
- `Main_*_Presentation.tex`: Beamer (`template_presentation.tex`)
- `Main_*_CheatSheet.tex`: two-column landscape article (`template_cheatsheet.tex`); seminars use
  `multicols{3}`, workshops `multicols{2}`

**Never use a float (`table[h]`, `figure`) in a content file that feeds a CheatSheet.** Floats
cannot be placed inside `multicols` and are silently dropped: no error, no warning. Use
`\begin{center}` + a bare `tabular`. Only visible by rendering the PDF.

**Two-column `adjustbox`+`minipage` frames need `%` line-endings.** The side-by-side convention
(text ~0.55`\linewidth`, diagram ~0.4`\linewidth`) leaves no room for inter-line spaces. End those
lines with `%` (`\end{minipage}%`, `}%`, `\hfill%`, `\adjustbox{valign=t}{%`), or the right
minipage wraps below the left and runs off the slide; the only log signal is one `Overfull \vbox`.
Apply this to every new two-column frame and render the page to confirm.

`beamerarticle` in `template_cheatsheet.tex` lets Beamer frames compile in article mode. Both
templates load `upquote` after `listings` so code quotes render straight.

**Silent vertical overflow is common and never warns.** A TikZ diagram or a code block followed by
prose can overflow a frame or a narrow CheatSheet column with no `Overfull` warning. Render suspect
pages (`pdftoppm -png -r 100 -f N -l N`) and look. For diagrams under the two-column convention,
wrap in `\adjustbox{max width=\linewidth, max totalheight=<value>}`; width-only capping does not
stop vertical overflow.

`template_presentation.tex`'s `lstset` carries `belowskip=-15pt`, so body text directly after
`\end{lstlisting}` is pulled into the code box. Fold a short trailing sentence into the listing as
a final `#` comment line.

### Naming conventions
- Topic files: `<domain>_<topic>.tex` (e.g. `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Drivers: `Main_[Course|Seminar|Workshop]_<Subject>_[Presentation|CheatSheet].tex`
- Every Seminar and Workshop **must** have both a `_Presentation.tex` and `_CheatSheet.tex` driver
- `_Short` on a **driver**: shorter-duration variant sharing topic files with its parent via the
  `X.tex`/`X_short.tex` comment-sync pattern
- `_overview` on a **topic file**: deep standalone treatment outside the `X.tex`/`X_short.tex`
  relationship
- `_Overview` on a **driver**: minimal single-section seminar making an `_overview.tex` topic file
  independently reachable (no References section, by design)

### Sibling-file sync rule
Whenever a `.tex` file being edited has an `X.tex`/`X_short.tex` comment-sibling, read the sibling
too, even if the driver does not `\input` it. Any frame added, removed or materially edited in one
must have its comment/uncomment state mirrored in the other. **Count *live* (uncommented) frames,
not raw `\begin{frame}` occurrences, when judging a deck's size**; raw counts overstate it.

Default to the full (`X.tex`) sibling over `_short` unless there is a genuine duration reason. Check
both before assuming one is current.

**Always invoke `/upgrade-deck` itself rather than reconstructing its checklist from memory.**
Manual passes have silently skipped items such as the prose-dash sweep.

### Orphan-file tooling
`LaTeX/find_orphan_tex.py` finds `.tex` files not reachable via `\input`/`\include` from any
`Main_*.tex` driver (the reverse of the `latex-audit` skill's `inputs` mode). It excludes
`images/*.tex` and `backup`/`_backup`/`_retired`. Run it after any bulk rename or reorganization.

### Known issues
- **Casing drift on the `Main_Seminar_AI_For_*` drivers.** Disk names use `AI_For_...`; git's index
  uses `AI_for_...` for at least Educators. Harmless on Windows; would break a case-sensitive
  clone. Deprioritized.
- **`Main_Course_GenerativeAI_*` has never compiled end to end.** It is a pdflatex/Beamer
  large-deck instability (non-converging `Overfull \hbox` loop), not a bug in any frame. **Do not
  run mega-deck compiles (multi-workshop or full-course) to chase it.** Work at single-seminar or
  single-workshop scale.
- Re-run `latex-audit inputs` after any bulk rename to check `\input` resolution.

### Frame boilerplate
```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{frame}[fragile]\frametitle{Slide Title}
\begin{itemize}
\item Point one
\end{itemize}
\end{frame}
```
Section divider:
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
3. Put images in `LaTeX/images/`

## Course structure (reference)

- **Machine Learning** (40hr): `course_machinelearning_content.tex`, with `seminar_ml_*_content.tex`
  seminars.
- **Maths for ML** (24hr): `course_maths4ml_content.tex`; seminars named
  `Main_Seminar_MathsML_<Topic>_<Subtopic>_*`.
- **Python** (16hr): `course_python_content.tex`; Basic and Advanced workshops, also usable
  standalone.
- **Reinforcement Learning**: `workshop_reinforcementlearning_content.tex`; seminars named
  `Main_Seminar_ML_ReinforcementLearning_<SubTopic>_*`.
- **CareerInDataScience**: Full and Short variants sharing topic files via the comment-sync pattern.
- **MLCoEP** ("AI-ML for Mechanical Engineers", CoEP course): independent sessions, each
  `seminar_mlcoep_session_<N>_content.tex` with its own driver pair
  `Main_Seminar_MLCoEP_Session_<N>_<ShortName>_{Presentation,CheatSheet}.tex`. Drivers live in
  `LaTeX/_retired/mlcoep_session_drivers/` and remain the active verification workflow: copy the
  pair to `LaTeX/` root, compile, verify, copy the PDF to `Publications/Presentations/`, then
  re-retire. Optional cuts are `%`-commented with an "Optional" marker; restore by uncommenting.
  Offline data and scripts are in `Code/mlcoep/`. `ml_pca`, `ml_kmeans` and `ml_production` are
  shared with the generic seminar decks, so edits there affect both.
  `course_mlcoep_content.tex` chains all sessions into one combined driver; **never compile it
  without asking first** (20+ minutes, 350+ pages, never verified).
- `_overview.tex` topic files (`dnlp_intro_overview`, `data_intro_overview`, `dl_intro_overview`,
  `nlp_embedding_overview`) each have a minimal standalone `_Overview` seminar driver.

### Local LLM (Qwen3) as a Groq alternative
`ChatLlamaCpp` (langchain_community, in-process) with a local `Qwen3-1.7B-Q4_K_M.gguf` is a
drop-in for `ChatGroq` in plain multi-turn chat: see `Code/langchain/langchain_v1_models.py` and
`Code/omni-rag/agent.py`. Needs `llama-cpp-python>=0.3.34` and `/no_think` appended to the prompt.
`bind_tools()` with `tool_choice="auto"` does not work reliably at this size; a forced
`tool_choice` does. `Code/langchain/langchain_v1_createagent.py` stays on `ChatGroq` for this
reason. Opt-in, not a default.

## Code Directory

Each `Code/` subdirectory is one library or topic. There is no single build or test command; run
scripts per subdirectory.

### Environment setup (conda)
```bash
conda env create -f Code/<subdir>/environment.yml
conda activate <env-name>
```
Do not create `venv/` or `.venv/`.

### Subdirectory map

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
| Research Refs | `txt2cad/`, `txt2sql/` (docs only) |
| Other | `amd/` (AMD Academy materials), `chromeext/` (Chrome extension) |

`Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, model weights
(`*.bin`, `*.pt`, `*.safetensors`), `node_modules/`, `.pytest_cache/`, `.ruff_cache/`,
`.benchmarks/`.

### Sub-projects with their own config
- `Code/claudecode/MyWorkshop/` and `Code/claudecode/CadCamWorkshop/`: own `CLAUDE.md`
- `Code/langgraph/open_deep_research-langcahin-ai/`: own `CLAUDE.md` and `README.md`
- `Code/crewai/researcher/`: `pyproject.toml` + `uv.lock` (uv workflow)

### Security note
`Code/google-adk/my_agent/.env` is gitignored but holds a real `GOOGLE_API_KEY` on disk; rotate it
in Google Cloud Console.

## Test Suite

Python-script directories have a `test_*.py` runnable with `pytest` in the `genai` conda env.
```bash
conda activate genai
cd Code/<subdir>
python -m pytest test_*.py -v
```
All suites, from repo root:
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

Suites cover `chatbot-faqs/`, `chatbot-multimodal/`, `omni-rag/`, `parsing/`, `graphrag/`,
`google-adk/` and `agno/`.

### Test design notes
- No real API calls: LLM and embedding clients are mocked with `unittest.mock`.
- No model downloads: `transformers` model loading is patched at function level.
- The `google-adk` tests mock the `adk` package (not installed everywhere).
- The omni-rag `TestOmniIngestorStructure` tests skip if a `datasets` circular import occurs in a
  combined session; they pass in isolation.

### Environment notes
- `ragas 0.4.3` upgraded `openai` from 1.x to 2.x; verify `langchain-openai` compatibility if
  issues arise.
- A broken system-Python `faiss` install exists at
  `C:\Users\yoges\AppData\Roaming\Python\Python310\site-packages\faiss\` and conflicts if imported
  outside the conda env.
- `opendataloader-pdf` wraps a Java JAR: **Java 11+ must be on PATH** before any tutorial runs
  (`conda install -n genai -c conda-forge openjdk=11`). Tutorial 09 (OCR) also needs the hybrid
  backend: `opendataloader-pdf-hybrid --port 5002 --force-ocr`.

## Memory
Do not store, write, or update memory files in the global `~/.claude/projects/` directory unless
the user explicitly allows it in the current conversation.

## Git
Do not run any git commands. The user manages all git operations externally.
