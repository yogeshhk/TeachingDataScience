---
name: qt-lecture-pipeline
description: Runs the repeatable pipeline that turns one seminar of QT-01, QT-02 or QT-05 (a Part, Seminar, Lecture course under LaTeX/fdps/) from its study notes and content-plan row into authored lecture stubs, compiled decks, an LMS page set and a textbook section. Reads the Notes file and content plan, presents a sourcing plan and waits for approval, authors the stubs, compiles, waits for the user's own review comments, runs /upgrade-deck, stages the LMS pages, drafts the textbook sections, and lists the manual Studio steps. Use when the user asks to "author seminar X-Y of QT-0N", "develop the next QT-05 seminar", "build out these lectures", or calls /qt-lecture-pipeline. Quantum-specific: only for the TeachingQuantumTech repo. Not for QT-03 (use /qt03-notes-to-slides-lms) and not for QT-00.
---

# QT Lecture Pipeline (QT-01, QT-02, QT-05)

**Scope: quantum-specific.** This command is written for the TeachingQuantumTech repo and its folder
layout (`LaTeX/fdps/QT-0N/`, `LMS/QT-0N/`, `LaTeX/textbooks/QT-0N/`). Its name starts with `qt-` for that
reason. It is not a generic command, and it needs adapting before use on another repo.

You are running the repeatable, per-seminar pipeline for the three courses built on the QT-03 model:
a Part, Seminar, Lecture tree (`X-Y-Z`) generated from a content plan. A Seminar groups 2 to 4 lectures
and is one 90-minute FDP slot. A Lecture is one 45-minute semester lecture and is the authoring unit. The
same `X-Y-Z` codes name the deck stubs, the LMS units and the textbook sections.

The argument is a course and a seminar, for example `QT-05 3-1`. If the seminar is omitted, find the next
seminar whose lecture stubs are still empty, by checking each stub for frames. If the course is omitted,
ask which one.

This command does the LaTeX-side work, the LMS staging and the textbook drafting. The Open edX Studio
sync is always manual and is listed as a checklist at the end, never executed by Claude.

## The sync rule that governs everything

The deck stubs are the only source of truth. The LMS pages, the Booklet, the cheat sheets and the textbook
are derived. Every fix, from the author's review or from you, goes into the deck stub first and then flows
outward. Never fix a claim only in a derived copy. The repo `CLAUDE.md` states this rule too.

## Booklet refresh, every time a stub changes

Whenever this command (or anything else) adds or edits a deck stub of the course, finish by compiling the
course Booklet (`make_booklet.bat` in `LaTeX/fdps/QT-0N/`) and copying the PDF to
`C:\Users\yoges\Desktop\`, overwriting the old file. List the Desktop file and confirm its size and
timestamp before reporting. Compile the Booklet only, never the whole-course Presentation deck. This holds
at every stage below, including the fixes from the author's review.

## Review order

Every step runs on Sonnet. The order is fixed:

- Author and compile the stubs (Sonnet).
- The author reads the Booklet (and the decks) by hand and writes comments.
- Every correction goes into the deck stubs only.
- `/upgrade-deck` (Sonnet).
- Only then regenerate the derived copies (Sonnet): recompile the cheat sheets, lecture decks and Booklet
  (they `\input` the same stubs, so they follow the stubs automatically), then sync the LMS pages and the
  textbook sections.

Do not stage the LMS or draft the textbook before the deck has passed the author's
review. Any claim fixed later would then be fixed in four places. When a stub changes after the textbook
or the LMS exists, re-sync them: run `/qt-draft-textbook` for the book, update the LMS page, and report
the LMS and PDF changes separately.

## Which courses and how they differ

| Course | Depth | Hands-on |
|---|---|---|
| QT-01 Survey of Quantum Technologies | qualitative: ideas, applications, no heavy derivations | demos and simulator walkthroughs only |
| QT-02 Foundations of Quantum Technologies | rigorous: derivations and proofs where the course needs them | light NumPy or QuTiP demos |
| QT-05 Introduction to Quantum Computation | algorithmic: circuits, complexity, resource counts | Qiskit, with reference solutions in `code/` as in `QT-03/code/` |

Where a topic recurs across courses, use the overlap map in `LaTeX/fdps/TODO.md` and keep notation and
claims consistent with the other course's stub.

## Ground yourself first, every time

Read these, in this order, before planning anything:

- `LaTeX/fdps/QT-0N/README.md`: the structure table, the file naming, the authoring conventions.
- `LaTeX/fdps/QT-0N/TODO.md` and the cross-course `LaTeX/fdps/TODO.md`: decisions already made, the
  overlap map, open items. If the seminar is owned by someone other than the user, confirm first.
- `LaTeX/fdps/QT-03/README.md` ("Deck content checklist") and `LaTeX/fdps/QT-03/TODO.md` ("Rules"), as the
  README of each course directs, adapted to a theory course, plus `LaTeX/AUTHORING.md`. Take depth and
  shape from QT-03 stubs that are already authored.
- `LaTeX/fdps/QT-0N/docs/Notes QT-0N.md`: the blocks for this seminar's lectures. This is the scope brief:
  keywords, hands-on idea, references, videos and Material. Each file ends with a list of unverified
  items; carry those into the sourcing plan and do not state them as fact in a stub.
- `LaTeX/fdps/QT-0N/docs/QT-0N_Content_Plan.csv` (the twin of the xlsx): this seminar's rows, with Part,
  Seminar, Lecture, AICTE topic, sub-topics marked CORE, SUPPORTING or OPTIONAL, hands-on and the stub
  filename. Do not invent scope beyond what a row lists without flagging it.
- This seminar's stubs (`qt0N_<domain>_<slug>.tex`), the seminar file `seminar_qt0N_X_Y_content.tex`
  and the matching `LMS/QT-0N/` and `LaTeX/textbooks/QT-0N/` files, so you know what exists.
- The AICTE syllabus PDF in `docs/`, for anything that is on the edge of scope.

## Plan the sourcing, per lecture, and wait for approval

For every lecture in the seminar, decide and write down:

- **Local source first.** The seed material named in `LaTeX/fdps/TODO.md` for this course: the legacy
  ready reference deck, `fdps/QT-00/`, `fdps/QT-03/`, `fdps/NMR/`, `fdps/Qiskit/`, `seminars/` decks and
  the QCNP book in `Publications/Books/QCNP/`. Sibling repos are limited to the allowed list in
  `GitHub/CLAUDE.md`. Grep frame titles instead of trusting file names. Live-lecture clicker decks do not
  transplant: rewrite complete derivations or code, turn quiz framing into stated facts and exercises, and
  redraw every diagram as a static panel.
- **Web second**, for what local sources do not cover. Start from the Notes references. Verify every
  arXiv identifier through the arXiv API before citing it. Cite a video only when a page that lists it was
  fetched.
- **Diagram plan.** Most slides carry a diagram. For an algorithm, default to a step-by-step trace with 3
  to 4 stacked rows, each a full snapshot with the changed elements highlighted. For a circuit, draw it
  with TikZ or `quantikz` as the stubs already do. Stack a reduction chain vertically, never horizontally,
  and put a boxed legend clear of the curves.
- **Exercise plan.** One frame titled `Exercise: <topic>` per lecture, with Routine, Applied and
  Challenge tiers. For QT-05 also plan the reference solution script for `code/`.
- **Depth.** Match the depth in the table above. For a 3-lecture seminar the FDP slot is condensed, so
  mark which lectures lose which frames in a Seminar delivery, but keep every lecture stub complete.
- **Overlap.** Name any topic that is also in another course and check that stub first.

Present the plan as a table (lecture, source, what is adapted and what is new, diagram plan, hands-on) and
wait for the user's go-ahead before writing any file. This is a bulk operation. Do it even when an earlier
seminar was approved, because every seminar's sourcing differs.

Keep the approved plan as a checklist in the session scratchpad, one line per stub, so that it survives an
interrupted session. Do not write it into `TODO.md`, which holds only open actionable items and rules.

## Author the lecture stubs

Replace each empty stub with real content, in the order: idea and motivation, closed-form result or
derivation, algorithm or circuit, worked example with numbers, implementation or demo, diagrams, exercise.
Stubs hold `\begin{frame}` blocks and `\subsection` headings only; the seminar file owns the `\section`.
Rules that all apply:

- Every slide stands on its own. No chapter, part, seminar, session, lecture or course numbers in visible
  text, and no "next", "previous", "above" or "below". A seminar deck may be delivered without its
  neighbours.
- No visible placeholders, no title-card frame at the top of a stub.
- No em dashes or long dashes in generated content (`—`, `--`, `---`), outside `%` comments and code
  syntax. Grep every changed file before calling it done.
- Learner-facing text follows the six ASD-STE100 rules in `LaTeX/AUTHORING.md`: one term per concept,
  imperative single-action steps, about 20 words per bullet, no long noun clusters, no ambiguous `-ing`,
  keep articles. They do not apply to explanatory narrative.
- Do not add `\usepackage` to a stub. Do not override the `lstlisting` style per listing; use a
  two-column layout when code is wide. Wrap a `tabular` with more than 2 to 3 short columns in
  `\resizebox{\linewidth}{!}{...}`. New colours go in both templates.
- Images go in the course `images/`, referenced without a path, with no spaces in the filename.
- Do not state counts of content units that grow in any README or CLAUDE.md prose.
- Frame titles of the form `<Topic>: <second part>` keep the second part only when it adds information
  that the section context does not.
- Do not go beyond the AICTE syllabus for this course. A motivating aside is one short paragraph, not a
  frame. Do not add content that the course decisions excluded.
- Every number in a worked example comes from running it. Write a short script, run it, and take the
  numbers from the output. Do not trust a first-draft hand calculation, and do not quote a number you
  could not reproduce.
- Qiskit code is run in the `quantum` conda environment
  (`D:\Yogesh\anaconda3\envs\quantum\python.exe`). Say in the stub or the report when a snippet was not
  run, and why.
- A claim about a date, a record, a roadmap or a company product is time-sensitive. Verify it from a
  fetched source and keep it generic when you cannot.
- Verbose explanatory prose goes in `% LMS NARRATIVE` comment blocks beside the slides, not on them.

If a first draft is short of the depth of an authored QT-03 topic, source more real content before
padding. Report the gap instead of hiding it.

## Reference solutions

- QT-05: write one runnable script per lecture exercise in `LaTeX/fdps/QT-05/code/`, named
  `qt05_<domain>_<slug>_exercise_solution.py`, with assertions, then run it in the `quantum` environment
  and fix what fails.
- QT-02: put each light demo in `code/` the same way, only where the lecture has a demo.
- QT-01: a demo or simulator walkthrough only where a lecture names one.

## Compile and check

Compile only this seminar's drivers, with `texify -cp --pdf <driver>.tex`, from the course folder:
`Main_Seminar_QT0N_X_Y_{Presentation,CheatSheet}.tex`, and from inside `lectures/` the matching
`Main_Lecture_QT0N_X_Y_Z_{Presentation,CheatSheet}.tex` for each lecture. Never compile the whole-course
Booklet as a smoke test of one seminar.

- Run the `verify-pdf` skill on every compiled PDF and read every contact sheet. Look at every diagram and
  every table at higher resolution. Overflow and overlap are silent in the log.
- Grep the compile log for `Overfull \vbox` and fix each one. Leave the known template font warning alone.
- Check cheat sheets separately: no `media9` or `multimedia`, wide tables resized, nothing cut off.
- Re-derive every number in every diagram and worked example, as described above.

## Stop for the author's manual review

When the stubs are compiled, tell the user which PDFs are compiled and where, and ask them to review the
Booklet by hand (and the decks). The Booklet is the best view for a read-through. Wait for their comments.
Apply every comment as a real fix in the deck stub, never in a derived copy, recompile the drivers and
re-check. Repeat until the author is satisfied. Then, and only then, continue.

Do not run `/upgrade-deck` before this point.

Do not keep a review-staging tracker or copy script. Copy a PDF to a location only when the user names
one, and always tell the user which files were regenerated so that they know what to re-stage.

## Run `/upgrade-deck`

Run the full `/upgrade-deck` skill, scoped to this seminar's stubs and drivers, including the manual half.
Address what it flags, recompile and re-verify.

## Stage the LMS pages

The LMS follows the QT-03 way: thin pages, with the teaching content staying in the deck. Do this only
after the author's manual review is done.

- Copy each approved lecture Presentation PDF (not the cheat sheet) into `LMS/QT-0N/static/`.
- Fill in `LMS/QT-0N/qt0N_X_Y_Z__content.html` for each lecture: an `<h2>` with the lecture title only,
  an intro sentence, a "Topics in This Lecture" list, one iframe embedding the Presentation PDF, then a
  Watch block with one primary video and one or two secondary links. Take the video identifiers from the
  Notes file, and only the ones it marks as verified.
- Once every lecture of the Part has its page, write `qt0N_X__quiz.xml`: one graded quiz per Part,
  spanning every lecture of the Part, in the staging rules of `LMS/QT-03/TODO.md`.
- Do not touch a file in `LMS/` that is not part of this seminar's pages.

Flag the LMS changes and the regenerated PDFs separately in your report, so the user knows what to sync to
Studio and what to upload to `static/`.

## Draft the textbook sections

The book elaborates the stubs, one section per lecture and one chapter per Part, in
`LaTeX/textbooks/QT-0N/qt0N_book_<part>.tex`. Run `/qt-draft-textbook` for this step; it holds the full
rules. The summary:

- Write each section as book prose from its stub, with derivations, worked examples and figures, keeping
  the `% SOURCE: ... | synced: DATE` comment and setting the date.
- Fill in the chapter's Exercises (Routine, Applied, Challenge) and Further Reading, and add answers to the
  solutions appendix.
- Run `python LaTeX/textbooks/_scripts/check_book_drift.py` and report its result.
- Compile the interior driver and verify it as above. The book never carries a claim that the deck does
  not.

Do this only for the lectures whose stubs have passed the author's review.

## Manual steps for the user: list them, never run them

End the output with this checklist:

- Mirror `LMS/QT-0N/` into the `IHUB-QT-0N` folder with FreeFileSync (that folder is off limits to Claude)
  once the static PDFs and the HTML and XML are ready.
- Check the synced result in `IHUB-QT-0N`.
- Paste the HTML and XML into Open edX Studio, upload the PDFs to Files and Uploads, and check that the
  embed and the video block render.
- Create the Section, Subsection and Unit outline in Studio if it does not exist yet, as listed in
  `LMS/QT-0N/TODO.md`.

## Close out

Report in the reply, not in `TODO.md`: page counts of each compiled PDF, any lecture where no good source
existed and the content was designed from reasoning alone, the unverified items that you carried from the
Notes file and what you did with them, every judgment call, and which steps are done and which are open.
Add to a `TODO.md` only an item that is still open and actionable. Write no history log there.

## Guardrails

- Never run any git command.
- Author only inside `LaTeX/fdps/QT-0N/` until the LMS step. Touch `LMS/QT-0N/` and
  `LaTeX/textbooks/QT-0N/` only in their steps, and only after the decks are approved.
- Never touch `IHUB-QT-0N`, `ImpDocs/` or any folder that `GitHub/CLAUDE.md` prohibits.
- Never move, copy or restructure files across folders without a numbered plan and confirmation. The
  lecture list changes only through the content plan and `_scripts/gen_scaffold.py`. If a lecture is
  renamed there, rename its old stub by hand so that its content is kept.
- Do not create `.github/workflows/` or any CI.
- This is a large task. Consider forking the authoring of each lecture to keep the main session free of
  per-file tool noise, but keep the sourcing plan and its approval in the main thread, and paste the hard
  rules above into the fork's prompt, because a fork does not read `CLAUDE.md`.
- If the seminar is owned by someone else, confirm with the user before authoring it.
