# AI for X: common core (Part A) and domain part (Part B)

Design record for the "AI for X" seminar family. The cores and AI for Authors are built, and the existing decks
(Tech Leaders, All Tech, All Non-Tech, Educators, Business Leaders, Project Managers) include the cores.

## Goal

Every "AI for X" seminar is two parts.

- **Part A, the core:** what AI and generative AI are, how an LLM works and where it fails, and how to prompt
  (CRAFT). One version for technical audiences and one for non-technical audiences. Written once, included by
  every deck of that kind.
- **Part B, the domain part:** what this audience does with AI. Written per audience.

A new "AI for X" needs only its Part B files, one content file and two drivers (Presentation and CheatSheet).

## Structure

### Part A cores

| Core | Files | Used by |
|---|---|---|
| Non-tech | `ai_core_nontech_whatisai`, `_limits`, `_prompts` | Authors, Educators, All Non-Tech, Business Leaders, Project Managers |
| Tech | `ai_core_tech_whatisai`, `_limits`, `_prompts` (the same CRAFT frames with engineering examples, plus the mechanisms behind each limit) | Tech Leaders, All Tech |

The optional peanut butter sandwich demo (`llm_promptengg_sandwich`) is shared by several decks and is not part of
either core.

### Part B, per audience

One set of files named `ai_for_<x>_*.tex`, with a fixed skeleton so all decks feel alike:

1. Opening: why this matters to this audience (a few frames).
2. The work in stages, each with CRAFT prompt examples (the bulk). The stages differ by audience: a time flow for
   authors, a task flow for managers, a classroom flow for educators.
3. Responsible use specific to the audience.
4. Start-small action plan and references.

A section divider opens Part B in every AI-for-X deck (in Authors, the frame "AI for Authors").

### Content file and drivers

```
seminar_ai_for_<x>_content.tex
  \section{Core}      \input{ai_core_nontech_whatisai} ... \input{ai_core_nontech_prompts}
  \section{<x>}       \input{ai_for_<x>_...}
  \section{Wrap-up}   \input{ai_for_wrapup} \input{ai_refs}
Main_Seminar_AI_For_<X>_Presentation.tex   (about_me, content, thanks)
Main_Seminar_AI_For_<X>_CheatSheet.tex
```

Rules that apply (from `AUTHORING.md`): no em dashes; no floats in files that feed a CheatSheet; two-column
frames need `%` line endings; render and look at pages; compile one driver at a time and ask before compiling.

## AI for Authors (the test case)

Part B files: `ai_for_authors_intro`, `_pre` (before writing: ideas, research, outline, title options), `_during`
(the text is written by hand; allowed: diagram preparation and web searches; not allowed: AI-written content),
`_post` (editorial review, consistency, layout review with LaTeX, proposal and publisher emails, launch posts),
`_responsible` (manuscript confidentiality, copyright and proof of authorship, publisher AI policies, disclosure,
keeping your own voice) and `_wrapup`. Content file `seminar_ai_for_authors_content.tex`; drivers
`Main_Seminar_AI_For_Authors_{Presentation,CheatSheet}.tex`. The PDFs are in `Publications/Presentations` and
`GDrive/ToPrint/Talks`.

Guardrails: describe publisher AI policies in general terms and tell authors to read each publisher's own
submission guidelines; no invented statistics; no claims about named companies. The deck stays as it is (books,
articles and papers in one place, book examples; no paper-specific frames).

## Decisions (author)

- To trim a deck, comment frames in place; never delete them.
- No short sibling for now; add a `_short` later only if needed.
- Three kinds of file: tech core, non-tech core, and many domain-specific Part B sets.
- The sandwich demo is not in the Authors deck; it stays an optional add-on.
- Promo slides (optional, include in any seminar): `promo_book_airupe`, `promo_book_ai_sakal`,
  `promo_book_mentalmodels_sakal`, `promo_book_gametheory_sakal` (coming soon), `promo_book_gametheory_apress`, and
  one-slide `promo_paic` (the long `paic.tex` and `airupe.tex` stay for the older decks). `promo_macros.tex` shows a
  framed placeholder until the real cover or QR file exists in `images/` (`Cover_<Book>.jpg`, `QR_<Book>.png`).
  The Authors Presentation driver has them as commented `\input` lines.

## Risks

- The Educators CRAFT files are written for teachers; the generic core replaces education examples with neutral
  ones, and education examples stay in Educators' Part B.
- A shared core changes every deck that includes it, so a later fix to a core file needs a recompile of each
  deck that includes it.
- `Main_Course_GenerativeAI_*` never compiled; do not include the cores there.
