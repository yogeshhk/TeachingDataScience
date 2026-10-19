---
name: coep-session-pipeline
description: Runs the repeatable pipeline for one session of the MLCoEP course (AI-ML for Mechanical Engineers, 20 one-hour sessions at CoEP) in the TeachingDataScience repo. Grounds in the repo rules, plans sourcing and waits for approval, authors or revises the session's topic files with verified numbers and mechanical-engineering examples, compiles through the per-session drivers using the copy, compile, verify, re-retire procedure, waits for the author's own review, runs /upgrade-deck, and delivers the PDFs to the agreed folders. Use when the user asks to "develop MLCoEP session N", "revise the CoEP session on X", "build out the next CoEP session", or calls /coep-session-pipeline. COEP-specific: only for the TeachingDataScience repo. Not for the quantum courses (use /qt-lecture-pipeline).
---

# COEP Session Pipeline (MLCoEP, TeachingDataScience)

**Scope: COEP-specific.** This command is written for the MLCoEP course in the TeachingDataScience repo
(`D:\Yogesh\GitHub\TeachingDataScience\`). Its name starts with `coep-` for that reason. It is modelled on
the quantum course pipeline (`/qt-lecture-pipeline`), but the structure differs, so do not carry quantum
paths or conventions over.

The course is **AI-ML for Mechanical Engineers**: 20 sessions of one hour for final-year B.Tech Mechanical
students at the College of Engineering Pune, pairing Python and ML fundamentals with mechanical-engineering
applications. The sessions are: AI Overview, Python Overview, EDA and Data Preparation, Pandas,
Introduction to ML, Core ML Concepts, ML Workflow and Model Evaluation, Feature Selection, Linear
Regression, Logistic Regression, Decision Trees, Ensemble Methods and Random Forest, SVM, Naive Bayes, KNN,
K-Means, PCA, Titanic Capstone, MLOps and Deployment, and AI/ML Applications and Project Ideas for ME.

The argument is a session number (1 to 20) or its short name, for example `16` or `KMeans`. If it is
omitted, ask which session. Do not guess.

## How the course is built here

- `LaTeX/<domain>_<topic>.tex`: topic files, the authoring unit, written as Beamer frames. Shared with the
  other courses and workshops of this repo, so a change may reach more than this course.
- `LaTeX/seminar_mlcoep_session_N_content.tex`: the session aggregator, a `\section` plus `\input` lines.
  A case study that only this course uses sits in its own topic file (for example
  `ml_kmeans_cars_case_study.tex`).
- `LaTeX/course_mlcoep_content.tex` and `Main_Course_MLCoEP_{Presentation,CheatSheet}.tex`: the combined
  course. Never compile these (see below).
- `LaTeX/_retired/mlcoep_session_drivers/Main_Seminar_MLCoEP_Session_<N>_<Name>_{Presentation,CheatSheet}.tex`:
  the per-session drivers. The folder name says "retired", but these are the active way to compile one
  session at a time.
- `LaTeX/template_presentation.tex` and `template_cheatsheet.tex`: shared templates.
- `Code/mlcoep/` (README, environment, datasets, sessions) and `Code/ml/` (data such as `cars.csv`): the
  code and data behind the slides.

The topic files are the single source of truth. The Presentation, the CheatSheet, any `_short` variant and
the combined course all `\input` them, so a correction goes into the topic file and flows to every output
on the next compile.

## Ground yourself first, every time

Read, in this order:

- `TeachingDataScience/CLAUDE.md` and `AUTHORING.md`: the rules, the hierarchy, the naming, the known
  issues. Also `D:\Yogesh\GitHub\CLAUDE.md` for the MLCoEP delivery rules.
- The session aggregator and every topic file it inputs, and each topic file's `X_short.tex` sibling when
  one exists.
- `Code/mlcoep/README_AIML4ME_2026.md` for the course objectives and learning outcomes, and the code and
  datasets that the session uses.
- `LaTeX/coep_course_logistics.tex` only if the session touches logistics.
- A session that is already complete (for example a neighbouring session) for depth and style. Match it.

## Plan the sourcing, and wait for approval

For the session, write down per topic: what is new or changed, the source, the diagram plan and the
hands-on or case study.

- **Local first.** Existing topic files in `LaTeX/` (`ml_*`, `python_*`, `maths_*`), `Code/ml/` and
  `Code/mlcoep/` notebooks and data. Grep frame titles instead of trusting file names. Sibling decks are
  often live-lecture decks: adapt them, do not paste them.
- **Web second**, for what local sources do not cover: library documentation, a standard treatment,
  an engineering dataset. Cite only what you fetched.
- **Mechanical-engineering flavour.** Each session should have at least one example or dataset from
  mechanical engineering (bearing diagnostics, predictive maintenance, quality prediction, process
  optimisation, the cars data). Prefer data already in `Code/`.
- **One hour.** A session is one hour. Judge the depth against that and say which frames are core and which
  are optional.

Present the plan as a table and wait for the go-ahead before editing any file. This is a bulk operation.
Do not edit a shared topic file that other courses use without saying so first: name the other drivers
that `\input` it, and ask.

## Author the content

Rules, from the repo's own `CLAUDE.md` and `AUTHORING.md`:

- No em dashes or long dashes (`—`, `--`, `---`) in any `.tex` file outside `%` comments and listings. A
  hook blocks a literal em dash; a `--` only warns. Grep before you finish.
- Never use a float (`table[h]`, `figure`) in content that feeds a CheatSheet. Use `\begin{center}` and a
  bare `tabular`; floats are silently dropped inside `multicols`.
- Two-column `adjustbox` plus `minipage` frames need `%` line endings (`\end{minipage}%`, `}%`,
  `\hfill%`), or the right column wraps below the left.
- A code block followed by prose is pulled up by `belowskip=-15pt`; fold a short trailing sentence into
  the listing as a final `#` comment line.
- Wrap diagrams in `\adjustbox{max width=\linewidth, max totalheight=<value>}`; capping the width alone
  does not stop vertical overflow.
- When you edit `X.tex`, read `X_short.tex` and mirror the frame, in comment or live state. Count live
  (uncommented) frames, not raw `\begin{frame}`.
- Do not add `\usepackage` to a topic file. Images go in `LaTeX/images/`.
- Every number in a worked example comes from a run. Write a short script in the session scratchpad and run
  it in the conda environment defined by the course (`Code/mlcoep/environment.yml`, or the `genai`
  environment for the repo tests). Do not quote a result you did not reproduce, and say when a snippet was
  not run and why.
- Python only through conda, never `venv`. Never put a key or a secret in any file.
- Keep a bullet short. Put long explanation in the speaker narrative comments (`% LMS NARRATIVE` blocks), not
  on the slide.

## Compile and verify (the per-session procedure)

Compile one session at a time with its own drivers. Do **not** compile the combined course
(`Main_Course_MLCoEP_*`) or any multi-workshop deck; it has been killed after 16 minutes and more than
124 MB, and the repo forbids it without the author's say.

- Copy (do not move) the session's two drivers from `LaTeX/_retired/mlcoep_session_drivers/` into `LaTeX/`.
  They resolve the templates relative to their own folder.
- From `LaTeX/`, run `texify -cp --pdf <driver>.tex` for the Presentation and for the CheatSheet.
- Read the log for real errors. The closing-page QR footline overfull and the per-frame plus-3.98 pt
  navigation-bar overfull are known cosmetic artifacts; leave them.
- Run the `verify-pdf` skill on both PDFs and read every contact sheet. Zoom into every page with a new or
  changed diagram, table or code block. Overflow is silent in the log. Check the CheatSheet separately for
  dropped floats and cut-off columns.
- Re-derive every number in every diagram and worked example.
- When the checks are done, move the two drivers back into `_retired/mlcoep_session_drivers/` and delete
  all build litter (`.aux .log .nav .out .snm .toc .vrb`) and any PDF left beside the sources in `LaTeX/`.

## The author's review

Tell the user which PDFs are ready and where. Wait for their own read-through. Apply every comment as a
real fix in the topic file, mirror it to the `_short` sibling, recompile with the procedure above and
re-check. Repeat until the author is satisfied. A separate model-based factual review is not part of this
pipeline; the user asks for one by design when they want it.

## Run `/upgrade-deck`

Run the `/upgrade-deck` skill itself, scoped to the session's topic files, including its manual half. Do
not rebuild its checklist from memory. Address what it flags, recompile and re-verify.

## Deliver the PDFs

Follow `D:\Yogesh\GitHub\CLAUDE.md`:

- Sessions 16 to 20: after a significant change, compile both outputs. Put both the Presentation and the
  CheatSheet in `Desktop\MLCoEP_Presentations`, and the Presentation also in
  `D:\Yogesh\GitHub\Publications\Presentations`. When a session is done, remove it from the Desktop folder
  only.
- Other sessions: copy the Presentation PDF into `Publications\Presentations`, overwriting the stale copy,
  as for the earlier sessions. If you are unsure of the destination, ask.
- Leave no PDF beside the sources in `TeachingDataScience\LaTeX`. The repo is public and never commits
  PDFs.
- Keep no review-staging tracker, CSV or copy script. Copy a file only where the author names or the rules
  above name, and tell the author which files were regenerated.
- After any copy, list the destination and confirm size and timestamp before reporting success.

## Related files to flag, not change

- `Code/mlcoep/` notebooks, scripts and datasets that the session uses: if a slide change makes one stale,
  tell the author and ask.
- `COURSES.md` and `README.md`: only if the session list or titles changed.
- COEP has no LMS and no textbook folder. Do not create either.

## Close out

Report in the reply: the compiled page counts, any topic where no good source existed and the content
was designed from reasoning alone, every judgment call, shared topic files that other courses also use, and
which steps are done and which are open. Put an item in a `TODO.md` only when it is still open and
actionable, and write no history log there.

## Guardrails

- Never run any git command.
- Work only inside `TeachingDataScience` (and the delivery folders named above). Do not touch other repos
  except `Publications/Presentations` for delivery. Prohibited folders stay prohibited.
- Never compile the combined course or any mega-deck. Never use `venv`.
- Do not write memory files to the global `~/.claude/projects/` directory unless the author explicitly
  allows it.
- Edit files with the Edit and Write tools, not with shell heredocs or `sed`. For a longer script, write it
  to a file, then run it.
- This is a large task. Consider forking the authoring of each topic file to keep the main session free of
  per-file noise, but keep the plan and its approval in the main thread and paste these rules into the
  fork's prompt, because a fork does not read `CLAUDE.md`.
