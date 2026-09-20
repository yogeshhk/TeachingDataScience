# CLAUDE.md

Open-source Data Science teaching material: `LaTeX/` (Beamer decks and two-column cheat sheets) and `Code/`
(Python scripts and notebooks). Read [`AUTHORING.md`](AUTHORING.md) before authoring, compiling or testing
(build commands, the content hierarchy, naming, known issues, the `Code/` map, test suites).

## Rules that apply everywhere here
- **No em dashes** in user-visible markdown (`README.md`, `CONTRIBUTING.md`, `COURSES.md`,
  `Code/README.md`) or in any `.tex` file. A `PreToolUse` hook (`.claude/hooks/check-em-dash.ps1`) blocks a
  literal em dash in `.tex` and `.md` edits (except `CLAUDE.md`); a literal `--` or `---` gives a warning.
- **Compile with MikTeX** `texify -cp <driver>.tex` from `LaTeX/`. **Compiled PDFs and build litter are
  never committed** (the repo is public; `LaTeX/.gitignore` covers them). MLCoEP output goes where
  `D:\Yogesh\GitHub\CLAUDE.md` says.
- **Every seminar and workshop has both a `_Presentation.tex` and a `_CheatSheet.tex` driver** sharing one
  content file. Topic files are `<domain>_<topic>.tex`.
- **Never use a float (`table[h]`, `figure`) in content that feeds a CheatSheet:** floats are silently
  dropped inside `multicols`. Use `\begin{center}` and a bare `tabular`.
- **Two-column `adjustbox` plus `minipage` frames need `%` line-endings** (`\end{minipage}%`, `}%`, `\hfill%`)
  or the right column wraps below the left.
- **Render pages to PNG and look.** Vertical overflow, diagram overflow and text pulled into a code box
  (`belowskip=-15pt` after `lstlisting`) never warn in the log.
- **Count live (uncommented) frames**, not raw `\begin{frame}`. When editing an `X.tex`, read its
  `X_short.tex` sibling and mirror the change. **Always run `/upgrade-deck` itself**; do not rebuild its
  checklist from memory.
- **Do not run mega-deck compiles** (multi-workshop or full-course, including the combined
  `Main_Course_MLCoEP_*` driver) without asking. `Main_Course_GenerativeAI_*` has never compiled end to end.
- **Python uses conda only** (`environment.yml` per subdirectory); no `venv` or `.venv`. Tests run with
  `pytest` in the `genai` env and mock all API calls; Java 11+ must be on PATH for `opendataloader`.
- **Security:** `Code/google-adk/my_agent/.env` holds a real `GOOGLE_API_KEY` on disk (gitignored); rotate it.
- Do not write memory files to the global `~/.claude/projects/` directory unless the user explicitly allows it.
