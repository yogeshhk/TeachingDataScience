---
name: prep-mlcoep-session
description: Compiles, upgrades, and renames a single session of the ML CoEP course (LaTeX/course_mlcoep_content.tex) in isolation -- comments out the other 18 sessions, compiles both drivers via make.bat, runs /upgrade-deck scoped to just that session, recompiles, and renames the two output PDFs to Course_MLCoEP_<N>_<ShortName>_{Presentation,CheatSheet}.pdf, then restores the file to all-sessions-active. Argument is a session number or topic name; ask if not given. Use when the user asks to "prep session N of ML CoEP", "compile and upgrade the next CoEP session", or calls /prep-mlcoep-session.
---

# ML CoEP Session Pipeline

`course_mlcoep_content.tex` drives `Main_Course_ML_CoEP_{Presentation,CheatSheet}.tex` and chains
19 sessions directly (no workshop/seminar layer -- see `CLAUDE.md`'s "ML CoEP course session
rebalancing" note). To review/upgrade one session at a time you must isolate it by commenting out
the rest, compile, upgrade, recompile, rename, then restore.

This command has no dependency on and takes no action related to
`LaTeX/todo_mlcoep_session_verification.md` -- resolve everything directly from
`course_mlcoep_content.tex`.

## Step 0: Identify the target session

The argument is a session number or topic name/keyword. If no argument was given, ask the user
which session before doing anything else.

Resolve it by reading the `\section[ShortLabel]{Session N: Full Title}` headers directly in
`LaTeX/course_mlcoep_content.tex` -- match by session number, or by keyword against the short
label / full title. If the match is ambiguous, confirm the exact session number and title with the
user before editing any files.

## Step 1: Isolate the session

A session's block is everything from its `\section{}` line up to (but not including) the next
`\section{}` line -- **not** a single `\input{}`. Several sessions `\input` multiple files under
one `\section` (e.g. Session 3: `ml_eda_intro`, `data_preparation_short`, `ml_eda_endtoend_churn`;
also Sessions 7, 11, 14, 18, 19) -- treat the whole run of `\input` lines as one unit to keep live
or comment out together. Some blocks also contain lines already commented out by the author (e.g.
"-- excluded, available on disk if wanted later:" notes) -- leave those exactly as they are; only
toggle the live `\section{}`/`\input{}` lines.

Before commenting anything out, print the exact list of live `\input{...}` files found in the
target session's block (there may be one or several -- see above). This is a visible checkpoint so
it's obvious the whole block was picked up, not just the first `\input` line.

In `LaTeX/course_mlcoep_content.tex`, comment out every session's block **except** the target
session. Comment with `%` at the start of each line in the block -- do not delete or reorder
anything. Leave the `\appendix` / References / Datasets Used section at the bottom untouched (it's
small and applies regardless of which session is active).

## Step 2: Compile

From `LaTeX/`, run `make.bat` (loops `texify -cp` over `Main_Course_ML_CoEP_*.tex`). Check the
resulting `.log` files for errors, undefined references, or missing images before proceeding. Note:
every frame throws a pre-existing, harmless `Overfull \hbox` footer-overflow warning (the title is
wider than the footline box) -- this is a known repo-wide cosmetic issue, not a new bug; don't let
it block progress. Stop and report if the log shows a real error instead.

## Step 3: Upgrade

Invoke the `upgrade-deck` skill against `LaTeX/Main_Course_ML_CoEP_Presentation.tex`. Because the
other 18 sessions are commented out, its `\input` traversal naturally scopes the review to just the
live session's topic files -- no special-casing needed. Apply its edits to the underlying topic
`.tex` files (not to `course_mlcoep_content.tex`, unless a Task 3 restructuring finding specifically
requires a section-level change -- flag that to the user before applying it).

## Step 4: Recompile

Re-run `make.bat` so the PDFs reflect `/upgrade-deck`'s edits. Check the log again.

## Step 5: Rename outputs

Rename both PDFs, keeping the two forms as separate files:
- `Main_Course_ML_CoEP_Presentation.pdf` -> `Course_MLCoEP_<N>_<ShortName>_Presentation.pdf`
- `Main_Course_ML_CoEP_CheatSheet.pdf` -> `Course_MLCoEP_<N>_<ShortName>_CheatSheet.pdf`

`<ShortName>` is the session's full title (the text after `Session N:` in its `\section{}` line)
with spaces replaced by underscores. All 19 session titles are already kept to one or two words for
exactly this purpose (e.g. Session 3's title is `EDA DataPrep` -> `EDA_DataPrep`) -- no derivation
or abbreviation needed, just read the title and join it.

## Step 6: Restore

Uncomment the other 18 sessions back in `course_mlcoep_content.tex` so the file returns to its
normal all-sessions-active state. **Never end a run with the file partially commented** -- this is
a hard guardrail; "all 19 active" is the file's required resting state.

## Guardrails

- Only touch the target session's block plus the comment/uncomment toggling needed to isolate and
  restore it -- don't reorder, edit, or renumber other sessions.
- Confirm the resolved session number/title with the user before editing if there was any ambiguity
  in Step 0.
- If `/upgrade-deck` proposes edits to a file with an `X.tex`/`X_short.tex` comment-sibling, its own
  Step 4 sibling-sync guardrail applies -- don't bypass it just because this pipeline is driving it.
- Report the final PDF filenames and log status back to the user at the end of the run.
