---
name: latex-audit
description: Runs deterministic checks against LaTeX teaching/book repos (TeachingQuantumTech, TeachingDataScience, Publications) instead of reasoning about them by reading files. "frames" counts live (uncommented) vs raw \begin{frame} occurrences in one file or a whole directory, since a raw grep count is known to overcount by including commented-out frames; "inputs" transitively walks \input{}/\include{} chains from one or more driver files and flags any target with no matching .tex on disk; "headings" walks the same chain and flags candidate frame titles with manually-assigned numbering (Exercise 1, Lab 2, Step 3...) that makes reordering painful, or that redundantly repeat a word from the deck's own \title{}; "spacing" checks every template_cheatsheet.tex for the compact bullet-list fix (enumitem + \setlist{nosep}). All modes skip backup/_backup/_retired folders. Use this whenever asked to judge a deck's real size, find "how many live frames" in a file, check whether a driver (or a whole repo) has broken/unresolved \input chains, review frame-title numbering/redundancy, or confirm the compact-list template fix is applied everywhere -- before doing any of that by hand with Read/Grep.
---

# LaTeX Audit

A single PowerShell script, `latex-audit.ps1` (same directory as this file), replaces manual
routines that this user's repos have needed repeatedly. `frames`, `inputs`, and `spacing` are
pure pattern-matching -- no judgment required, so don't spend reasoning tokens re-deriving them.
`headings` is different: it surfaces **candidates** for two known issues, not verdicts -- see
its own section below before acting on its output.

## When to use this

- User asks "how many live frames does X have" / "is this deck too big" / wants a real size
  count, not a raw grep count. **Always use Live, never Raw, when reporting deck size** -- raw
  counts routinely overcount by 30-50% because these repos' authors leave large chunks
  commented out rather than deleted.
- Before or after any bulk rename/restructure of `.tex` files, to catch a broken `\input`/
  `\include` chain the same way the repo-wide audits in these repos' CLAUDE.md history did.
- As a quick sanity check before running `/upgrade-deck` on a driver, to confirm its full
  chain actually resolves before doing a deeper content review. `/upgrade-deck` runs `inputs`
  and `headings` on the target driver as part of its own Step 1 -- see that command file.
- User asks you to check a deck (or the whole repo) for frame titles with painful manual
  numbering, or titles that redundantly restate the deck's own subject -- run `headings` first
  instead of re-deriving the pattern by reading every file, then apply judgment to what it
  flags.
- User asks whether the compact bullet-list template fix (no vertical gap between itemize/
  enumerate items, added 2026-08-25) has made it to every `template_cheatsheet.tex` copy across
  these repos -- run `spacing` instead of grepping each copy by hand.

## How to run it

```powershell
powershell.exe -NoProfile -File "C:\Users\yoges\.claude\skills\latex-audit\latex-audit.ps1" -Mode frames -Path <file-or-directory>
powershell.exe -NoProfile -File "C:\Users\yoges\.claude\skills\latex-audit\latex-audit.ps1" -Mode inputs -Path <file-or-directory>
powershell.exe -NoProfile -File "C:\Users\yoges\.claude\skills\latex-audit\latex-audit.ps1" -Mode headings -Path <file-or-directory>
powershell.exe -NoProfile -File "C:\Users\yoges\.claude\skills\latex-audit\latex-audit.ps1" -Mode spacing -Path <file-or-directory>
```

- `-Mode frames -Path <file>`: live/raw frame count for that one file.
- `-Mode frames -Path <directory>`: recurses over every `.tex` file, prints a per-file line
  plus a grand total. Use this on a whole family folder (e.g. `LaTeX\qcnp\workshops\`) or a
  whole repo's `LaTeX\` root.
- `-Mode inputs -Path <driver.tex>`: walks that one driver's full `\input`/`\include` chain,
  reports every unresolved target (with the file that referenced it and the path it expected).
  Zero output after the summary line means fully resolved.
- `-Mode inputs -Path <directory>`: finds every `Main_*.tex` file recursively and treats each
  as an independent driver, aggregating a `BLOCKED: ...` block per broken driver. This is the
  repo-wide sweep shape.
- `-Mode headings -Path <driver.tex>`: extracts the driver's `\title{...}`, walks its full
  `\input`/`\include` chain, and prints two candidate lists across every `\frametitle{...}` (and
  short-form `\begin{frame}{...}`) it finds: possible reorder-hostile numbering, and possible
  redundant subject naming (with which title word(s) triggered the match).
- `-Mode headings -Path <directory>`: same, run independently over every `Main_*.tex` driver
  found recursively.
- `-Mode spacing -Path <file>`: checks that one `template_cheatsheet.tex` has both `enumitem`
  loaded and a `\setlist{...}` containing `nosep`, uncommented.
- `-Mode spacing -Path <directory>`: recurses for every file literally named
  `template_cheatsheet.tex` and reports `OK`/`MISSING` per copy, plus a total.

All modes silently skip anything under a directory literally named `backup`, `_backup`, or
`_retired` -- these repos' own convention for dead/archived content that was never meant to
compile. If a repo turns out to have another such directory under a different name, that's
worth surfacing to the user rather than silently expanding this list.

## Interpreting results

- `frames`: the `<- N commented out` suffix on a line means Raw and Live differ. Report Live as
  the real count; mention the delta only if it's large enough to matter (a handful of quiz
  frames toggled on/off isn't noteworthy; 50+ commented frames worth flagging as "raw count is
  misleading here").
- `inputs`: a target resolves relative to the **including file's own directory** (these repos
  keep each family folder self-contained -- a topic file is always a sibling of the driver/
  content file that `\input`s it), and a target already ending in `.tex` in the source
  (e.g. `\input{images/tikz/array.tex}`) is used as-is, not double-suffixed. If something shows
  as blocked, don't assume it's a real bug -- open the referencing file and confirm before
  reporting it as broken, the same way you would with a Grep result.
- `headings`: **both lists are candidates for human review, not findings to apply blind.**
  Known legitimate exceptions the script cannot tell apart from real issues:
  - Canonical/externally-fixed sequences (QT02's "Postulate 1"-"Postulate 7", certification's
    "Section 1"-"Section 8" and "Task X.Y" exam-blueprint numbers) should usually stay numbered
    -- reordering them would break the sequence's own logic, not just the label.
    An algorithmically-fixed two-step structure (e.g. "Step 1: The Oracle" / "Step 2: The
    Diffusion Operator" in a Grover's-algorithm frame pair) is a closer call than an arbitrary
    author-assigned sequence like "Lab Exercise 1/2/3/4" -- weigh whether the number reflects an
    inherent order or just an author's pick.
  - A frame title naming a second framework/product by contrast (e.g. explaining Qiskit next to
    Cirq) is not "redundant subject naming" even if it matches a stopword-filtered title word.
  - The subject-word extraction is a heuristic (strip a fixed stopword list, take the phrase
    after the last `:` or before the first ` for `) -- expect real false positives, and don't
    expand the stopword list reflexively just to silence one; confirm the frame is actually
    restating the deck's own subject before proposing a change.
- `spacing`: `MISSING` means the compact-list fix (added to the shared `cheatsheets/`,
  `qcnp/workshops/`, `certification/`, `openedx/` templates plus the `Publications/`,
  `TeachingDataScience/`, and `ReadyRefLaTeX/` copies on 2026-08-25) hasn't reached that copy
  yet. `backup/aicte/`'s copy is intentionally never touched (read-only archive) and won't show
  as a problem to fix.

## Known limitations (don't silently over-trust this)

- Resolution is same-directory-relative only. A repo convention that puts topic files in a
  different subdirectory than their driver would show false positives here.
- Frame counting is line-anchored (`\begin{frame}` and a leading `%` on the same line). A frame
  commented out via some other mechanism (a block comment, `\iffalse`) won't be caught as
  commented, and would inflate Live incorrectly. This hasn't shown up in these repos so far.
- `headings`' title/frametitle extraction is also line-anchored and single-line: a `\title{...}`
  or `\frametitle{...}` split across multiple lines won't be picked up.
