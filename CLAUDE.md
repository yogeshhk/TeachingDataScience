# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

An open-source educational repository containing:
- **LaTeX**: Beamer presentation slides and two-column cheatsheets for Data Science courses (Python, ML, DL, NLP, GenAI, RAG, etc.)
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

Every deliverable has two output forms sharing the same content:
- `Main_*_Presentation.tex` — Beamer slides (`\documentclass{beamer}`, uses `template_presentation.tex`)
- `Main_*_CheatSheet.tex` — Two-column landscape notes (`\documentclass{article}`, uses `template_cheatsheet.tex`)

Both driver files `\input{}` a shared `*_content.tex` file (e.g., `seminar_artificialintelligence_tools_content.tex`) which in turn `\input{}`s individual topic files (e.g., `ai_tools_claudecode_intro.tex`).

### Naming conventions
- Topic files: `<domain>_<topic>.tex` (e.g., `maths_linearalgebra_matrices.tex`)
- Content aggregators: `<type>_<subject>_content.tex`
- Driver files: `Main_[Course|Seminar|Workshop]_<Subject>_[Presentation|CheatSheet].tex`
  - Seminar ≈ 1 hour, Workshop ≈ 1 day, Course ≈ 1 week/semester

### Adding a new topic
1. Create `LaTeX/<domain>_<topic>.tex` with Beamer frames
2. `\input{<domain>_<topic>}` inside the relevant `*_content.tex`
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
| Document Parsing | `docling/` |
| Deep Learning | `keras/`, `dl_tf2/`, `pytorch/`, `deep_rl/` |
| Classical ML | `ml/`, `math/`, `python/` |
| NLP | `nlp/`, `dnlp/`, `spacy/` |
| GNN | `pyg/` |
| Indic Language | `mahamarathi/`, `sarvam/`, `orgpedia/` |
| Research Refs | `txt2cad/`, `txt2sql/` (docs only, no runnable code) |

### Code/.gitignore
A repo-wide `Code/.gitignore` covers `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.pyc`, model weights (`*.bin`, `*.pt`, `*.safetensors`).

### Notable sub-projects with their own config
- `Code/claudecode/` — has its own `CLAUDE.md`
- `Code/langgraph/open_deep_research-langcahin-ai/` — has its own `CLAUDE.md` and `README.md`
- `Code/crewai/researcher/` — uses `pyproject.toml` + `uv.lock` (modern uv workflow)

### Security note
`Code/google-adk/my_agent/.env` is gitignored but contains a real `GOOGLE_API_KEY` on disk — rotate it in Google Cloud Console.

## Git
Do not run any git commands. The user manages all git operations externally.
