# /proof-read-chapters

Perform a final editorial review of book chapter(s).

## Input
`$ARGUMENTS` can be:
- A **single file path** — e.g., `Ch01_What is Game Theory.pdf`
- A **folder path** — process all files found in that folder
- A **glob pattern** — e.g., `*.pdf`, process all matching files
- An optional mode flag (anywhere in the argument string):
  - `--comprehensive` or `-c` → run all three tasks including deep copyediting
  - *(no flag)* → default Content Review mode (faster; skips copyediting)

**Examples:**
```
/proof-read-chapters Ch01.pdf                    ← default mode
/proof-read-chapters Ch01.pdf --comprehensive    ← comprehensive mode
/proof-read-chapters *.pdf -c                    ← comprehensive, short flag
```

---

## Modes

### Default mode — Content Review
Runs **Tasks 2 and 3 only** (Proofreading + Redundancy). Skips line-level grammar and punctuation checks. Faster; focuses on meaning-level problems: wrong wording, awkward logic, and duplicate explanatory content.

### Comprehensive mode (`--comprehensive` / `-c`)
Runs **all three tasks**: Copyediting + Proofreading + Redundancy. Use when a chapter needs a full grammar and punctuation pass in addition to content analysis.

**Always print a one-line mode banner at the top of each chapter's review:**
> `Mode: Content Review (default) — copyediting skipped`
> `Mode: Comprehensive — copyediting + content review`

---

## Process
- Parse mode flag from `$ARGUMENTS` first; strip it before treating the remainder as the file/folder/glob target
- Process files **one at a time**
- After presenting findings for each file, **wait for the user to confirm** ("yes", "ready", "next", or equivalent) before moving to the next file
- Before starting, check the progress memory file (see Progress Tracking below) to determine which files are already done and resume from where the session left off

---

## Tasks

### Task 1 — Copyediting *(Comprehensive mode only)*
Fix grammar, punctuation, and syntax errors. Catch:
- Comma splices
- Missing commas after introductory phrases or before coordinating conjunctions (only when genuinely ambiguous or incorrect)
- Subject-verb agreement errors
- Broken sentence structure — missing words, misplaced semicolons
- Missing end-of-sentence punctuation
- Incorrect or missing hyphenation
- Spurious quotation marks around standard noun phrases
- Subject/person inconsistency within a passage
- Parallel structure violations in lists or series

### Task 2 — Proofreading *(Both modes)*
Flag clarity, word choice, and logic issues:
- Vague or weak sentence openers that undermine otherwise crisp prose
- Illogical or incoherent phrases
- Subject shifts mid-paragraph
- Redundant phrasing within a sentence (e.g., "their own behavior by themselves")
- Awkward constructions (e.g., "better than in competitive examinations")
- Word choice errors — wrong word used (e.g., "intentions" where "incentives" is meant)

### Task 3 — Paragraph-level redundancy check *(Both modes)*
Identify paragraphs that state the **same concept in different words** — pure explanation with no distinct example content. This is an optional, high-bar task: only flag a pair when you are confident both paragraphs are genuinely interchangeable in meaning.

**Do NOT flag if:**
- Either paragraph contains an example (numerical, scenario, or domain-specific illustration), even if the surrounding concept is similar
- The paragraphs draw examples from different domains (e.g., one uses economics, another uses biology) — multi-domain illustration is intentional
- The distinction could be intentional emphasis or scaffolding

**Do flag if:**
- Both paragraphs contain only abstract/conceptual explanation (no examples) AND say the same thing in different words

After flagging, provide a merged replacement ready to copy-paste.

---

## Output format per chapter

### Section A — Copyediting *(Comprehensive mode only; omit entirely in default mode)*
One entry per issue, no preamble:

**"[first 4–6 words of paragraph]…"**
> [quoted problematic text]
Fix: [corrected text]

### Section B — Proofreading *(Both modes)*
One entry per issue, no preamble. In default mode this is renamed **Section A — Proofreading**:

**"[first 4–6 words of paragraph]…"**
> [quoted problematic text]
Fix: [corrected text]

### Section C — Paragraph-level Redundancy *(Both modes)*
In default mode this is relabeled **Section B — Redundancy**. Only include if redundant pairs were actually found. If none, write: *No redundancy candidates found.*

For each pair:

**A:** "[first 5–7 words]…"
**B:** "[first 5–7 words]…"
**Action:** DELETE paragraph B. Replace paragraph A with:
[merged paragraph — plain prose, no block formatting]

### Summary Table
`# | Location | Type | Fix summary | Severity`

The `Type` column values in default mode will be `Proofreading` or `Redundancy` only.

---

## Ground rules
- Suggest **critical changes only** — do not nitpick regional spelling, stylistic choices, or authorial voice
- Do not rewrite passages unless fixing a flagged issue or providing a redundancy merge
- Preserve the author's tone, rhythm, and register throughout
- Do not add praise or commentary about strong passages — focus only on what needs fixing
- When providing merged content for redundant paragraphs, match the author's existing style exactly

---

## Progress tracking
After delivering each chapter's review and receiving user confirmation, update the memory file:

**Memory file location:**
`C:\Users\yoges\.claude\projects\D--Yogesh-GDrive-ImpDocs-Publications-Strategy-ApressChapters-TechnicalReviewIncorporated\memory\copyediting_progress.md`

Log per chapter: chapter name, page count, mode used, issues found (brief list), redundancies found.

On session start, read this file first to determine what has already been reviewed and resume from the next pending chapter. If a chapter was reviewed in default mode, note it so the user knows a comprehensive pass is still available if needed.
