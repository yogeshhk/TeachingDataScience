---
name: qt-draft-textbook
description: Drafts or re-syncs a textbook chapter under LaTeX/textbooks/QT-0N/ (QT-01, QT-02, QT-03 or QT-05) from that course's deck stubs in LaTeX/fdps/, which are the single source of truth. Turns terse Beamer frames into explanatory book prose with verified numbers, figures, code, graded exercises and further reading, then updates the solutions appendix, compiles, checks the pages and runs the drift script. Acts as an updater on a section that already has prose, never a blind recreator. Use when the user asks to "draft the book chapter for X", "write the textbook section from the slides", "re-sync the book with the decks", or calls /qt-draft-textbook. Quantum-specific: only for the TeachingQuantumTech repo. Not for the QCNP book (use /qcnp-draft-chapter).
---

# QT Textbook Drafting

**Scope: quantum-specific.** This command is written for the TeachingQuantumTech repo and its folder
layout (`LaTeX/fdps/QT-0N/`, `LaTeX/textbooks/QT-0N/`). Its name starts with `qt-` for that reason. It is
not a generic command, and it needs adapting before use on another repo.

You are turning the lecture or topic stubs of one course (`LaTeX/fdps/QT-0N/`) into book prose in
`LaTeX/textbooks/QT-0N/`. The stubs are terse Beamer frames written to be projected. The book is read from
start to finish by a student alone with the page. Same claims, much more explanation.

The argument is a course and a chapter, for example `QT-05 Fourier Algorithms` or `QT-03 probstat`. For
QT-01, QT-02 and QT-05 a chapter is one Part and a section is one lecture. For QT-03 a chapter is one AICTE
chapter and a section is one topic stub. If the chapter is omitted, find the next chapter whose sections
are still `synced: never`, using `python LaTeX/textbooks/_scripts/check_book_drift.py -v`.

## The sync rule

The deck stubs are the only source of truth. The book is derived from them. Where the book and a stub
disagree, the stub wins. If you find an error or a gap in a stub while drafting, do not fix it only in the
book. Tell the user, and with their confirmation fix the stub first, then the book. Never let the book
carry a claim that the deck does not. Do not edit an LMS page or a Booklet in this command.

## If you change a stub

This command works from the stubs and does not change them. If, with the user's confirmation, you do fix a
stub, finish by compiling that course's Booklet (`make_booklet.bat` in `LaTeX/fdps/QT-0N/`) and copying the
PDF to `C:\Users\yoges\Desktop\`, overwriting the old file. Confirm size and timestamp. Never compile the
whole-course Presentation deck.

## When to run this command

Draft a chapter only after the author's manual review of the Booklet is done. Before that point a fix would have to be made in four places. If the user asks for a draft
earlier, say so, and mark the sections in your report as "drafted before review: expect a re-sync".

After a deck change, run this command again on the affected chapter. Corrections from the author arrive as
stub changes; the drift script lists the sections that went stale. Re-sync those sections as an updater,
set the new `synced` date, and report them. This command adds no new facts of its own.

## Ground yourself first, every time

Read, in this order:

- `LaTeX/textbooks/README.md` and `LaTeX/textbooks/TODO.md`: layout, the source line convention, open items.
- `LaTeX/textbooks/QT-0N/template_book.tex`: which environments and listing style exist, so that you use
  them and add nothing. Do not add `\usepackage` calls to a chapter file.
- One chapter of the same book that is already drafted, or of `QT-03` if none is, for voice, depth and
  layout. Match it.
- The chapter file to be drafted, and its `% SOURCE:` lines.
- Each source stub, with the TikZ and `\node` lines skimmed, since the diagrams are redrawn for the book.
  Note every claim, number and code snippet in it.
- `LaTeX/fdps/QT-0N/docs/Notes QT-0N.md`, for the references and the list of unverified items; carry those
  items into the book only as flagged, never as fact.
- `LaTeX/fdps/QT-0N/code/` for the reference solutions, where it exists.
- The course content plan row for each lecture, for the sub-topic scope and the intended depth.

Depth by course: QT-01 qualitative, QT-02 rigorous with derivations, QT-05 algorithmic with circuits and
resource counts, QT-03 numerical with code.

## Update or create: read the target first

- A section that is still the scaffold (comment lines only, `synced: never`): write it fresh.
- A section that already has prose: you are an updater. Compare it with its stub. Keep the author's voice,
  examples and structure where the concept has not changed. Add what the stub now has and the section
  lacks. Flag anything in the section that the stub does not support. Never regenerate a section just
  because the command ran again.
- A stub that has no section in the chapter (the drift script lists it as UNCOVERED): add a section for it
  in the order of the seminar file, by hand.

## Verify before you write

Every number printed in the book must come from a run. For each section, write a short check script in
the session scratchpad and run it in the `quantum` conda environment
(`D:\Yogesh\anaconda3\envs\quantum\python.exe`). The script does three jobs:

- Reproduce every number, output and timing that the stub states, and confirm that they agree. If one
  does not, stop and tell the user: the stub has an error, and the fix goes into the stub first.
- Produce the numbers for any new worked example that you add.
- Generate the figures.

Rules for this step:

- Run a library snippet before saying that the library is unavailable or that a snippet was not run. Check
  with an import first. If it really cannot run, say so in the text and in your report.
- Timings come from one machine. Write "on one test machine" and never promise a number.
- Keep the script in the repo as `LaTeX/textbooks/QT-0N/_scripts/chN_figures_and_checks.py`, one per
  chapter, as QT-03 does. Save each PNG into `LaTeX/textbooks/QT-0N/images/` as `qt0n_fig_<topic>.png`,
  lowercase, no spaces, no chapter or section number. Write the script in the scratchpad first, run it, and
  copy it into `_scripts/` once the chapter is done.
- Quote a date, a record or a product claim only from a source you fetched. Keep it generic otherwise.

## What a section needs

Write book prose, not frames turned into sentences. In the chapter file:

- A short opening before the first section: what the chapter covers and why, in a few sentences.
- One `\section` per lecture or topic stub, in stub order, each with its `% SOURCE:` line. Inside it,
  `\subsection*` headings. No heading numbers.
- The idea first, then the closed form or derivation, then the algorithm or circuit, then a worked example
  with real numbers, then code. Follow every equation with a plain sentence saying what it means.
- Explain every step of a worked example in prose. Do not paste the block and move on.
- Code at the levels that the stub has (a library call and a from-scratch version). Print the real output
  in a comment or in the text. Keep listing lines short, about 70 characters, since a long line wraps and
  hurts reading.
- A figure wherever a reader alone on a page needs one. Redraw the stub's diagram as a clean figure, or
  generate a plot from your check script. Put wide figures at `0.7\linewidth` or less with `keepaspectratio`.
- Do not state a statement as settled when the stub marks it optional or the Notes file marks it
  unverified.

Then, at the end of the chapter:

- `\section*{Exercises}`: one `\paragraph{<topic>.}` per lecture or topic, each with Routine, Applied and
  Challenge items in `\textbf{...}` run-in form, taken from the stub's Exercise frame and extended where it
  is thin. Solutions never go here.
- `\section*{Further Reading}`: a `itemize` of real sources, each named with author or title and, for
  online sources, a URL that you checked. No placeholder entries.

Learner-facing rules: the ASD-STE100 word limits apply to slides and instructions, not to explanatory book
narrative. Keep a bullet or a procedure step short and imperative, and let the narrative breathe.

## Style rules

- No em dashes or long dashes (`—`, `--`, `---`) in prose. TikZ, code and `%` comments are exempt. Grep the
  chapter before calling it done. The PDF check will flag minus signs and code hyphens as dash-like; those
  are false alarms, so read the source grep, not the flag.
- No "Chapter N", "Section N", lecture or session numbers in the text. Refer to other material by name, as
  "the linear algebra chapter".
- Avoid a long `\texttt{...}` identifier with underscores in running text; it cannot break and overflows
  the margin. Say "the function of this section" or move the name into a listing.
- Do not state counts of units that grow (seminars, lectures, chapters) in prose.
- Run a `/make-anti-ai` style pass over the drafted prose: uniform sentence length, one-line filler
  paragraphs, rhetorical triads, confident claims without a source. Fix them in place.
- Use exact and modest wording. Do not claim that something is the best, the first or the only without a
  source.

## Solutions appendix and code appendix

- Add an answer for every exercise group to `qt0n_book_solutions.tex`, under the chapter's `\section*`.
  State the key result with its number, and explain a conceptual answer in a few sentences. Take numbers
  from running the reference solution or your check script. Say when a value depends on random numbers or
  timing, and give the pattern to check.
- QT-03 (and any course that gets a `code/` folder): the appendix `qt0n_book_code_solutions.tex` is
  generated, one `\lstinputlisting` per solution script, in book order. Edit the `.py` files and
  regenerate; do not hand-edit it. Running the scripts rewrites any PNG that they save beside themselves,
  so tell the user.
- A new chapter's front matter (Preface, How to Use, Notation) and the references appendix are written once
  the first chapters exist. Do them only when asked or when the book is nearly complete.

## Compile and check

- Compile the interior only: `texify -cp --pdf Main_Book_QT0N_interior.tex` from `LaTeX/textbooks/QT-0N/`.
  The cover build is separate.
- Read the log for errors, undefined references and `Overfull` boxes that are worse than a point or two.
- Run the `verify-pdf` skill on the PDF. Read every contact sheet for the new pages, then zoom into any
  page with a figure, a table or a wide listing. Fix text that runs off the margin, a listing that wraps
  badly, and a table that is cut off.
- Set `synced: YYYY-MM-DD` on every section that you wrote or re-synced, then run
  `python LaTeX/textbooks/_scripts/check_book_drift.py` and report its line for this course. It must show
  0 stale and 0 missing sources for what you touched.

## Output

Report, in this order:

- Which stubs were read, which chapter file this maps to, and whether each section was created or updated.
- For an update, a short list of what changed and why.
- Every place where a stub and a run disagreed, or where a stub looked thin or wrong. These go to the user,
  not into silent fixes.
- Unverified items from the Notes file that you left flagged or left out.
- The figures created, the page count, and the drift script result.
- What is still open: sections not drafted, exercises without answers.

Update `LaTeX/textbooks/TODO.md` only for an item that is still open and actionable. Write no history there.

## Guardrails

- Never run any git command.
- Work only inside `LaTeX/textbooks/QT-0N/`. Read the stubs and the code folder; change a stub only with the
  user's confirmation, and then list that change separately so that the LMS and the Booklet can follow.
- Do not overwrite existing section prose wholesale.
- Do not add packages, macros or environments that the template does not define; flag the need instead.
- Do not guess a chapter or section mapping. Use the structure of the book and the `% SOURCE:` lines, or
  ask.
- Edit files with the Edit and Write tools, not with shell heredocs or `sed`. For a longer script, write it
  to a file, then run it.
- If the lecture stubs of a chapter are still empty scaffolds, stop and say so. There is nothing to
  elaborate yet; the lecture pipeline (`/qt-lecture-pipeline`) comes first.
- This is a large task for a long chapter. Consider forking one section at a time to keep the main
  session free of per-file noise, and paste these rules into the fork's prompt, because a fork does not
  read `CLAUDE.md`.
