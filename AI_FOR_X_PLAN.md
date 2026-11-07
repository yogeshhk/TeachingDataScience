# AI for X: common core (Part A) and domain part (Part B)

Plan and analysis, 2026-11-07. Nothing in the existing decks has been changed. Decisions by the author so far:
AI for Authors uses the non-tech core; existing decks stay untouched until AI for Authors is built and seen.

## 1. The goal

Every "AI for X" seminar is two parts.

- **Part A, the core:** what AI and generative AI are, how an LLM works and where it fails, and how to prompt
  (CRAFT). One version for technical audiences and one for non-technical audiences. Written once, included by
  every deck of that kind.
- **Part B, the domain part:** what this audience does with AI. Written per audience.

A new "AI for X" then needs only its Part B files, one content file and two drivers (Presentation and CheatSheet).

## 2. What exists today (measured with latex-audit, live frames)

| Deck | Live frames | How it is built | Own "what is AI" opening? |
|---|---|---|---|
| AI for Educators | about 65 | `ai_educators_content` (16), `llm_promptengg_educators` (15), `llm_promptengg_sandwich` (14), `ai_educators_wrapup` (13), refs, institution appendix | Yes, 4 frames, then CRAFT. The best model of the pattern |
| AI for Educators, Flipped | about 70 | One 541-line content file with CRAFT and exercises inline | Partly; CRAFT is copied in |
| AI for All, Non-Tech | about 109 | `ai_intro_nontech` (43), `ai_fluency` (11), ChatGPT short (8), applications (11), sandwich (14), mid-career (9), conclusion (12) | Yes, 43 frames, everyday-life tone |
| AI for All, Tech | about 159 | `ai_intro_tech` (45 live, 19 commented), healthcare, impact, `llm_fromzero` (31), ChatGPT (17), applications, mechanical-engineering applications, sandwich | Yes, long history of AI and hype cycle |
| AI for Business Leaders | about 51 | 7 files: intro, demystifying (13), possible, hands-on, strategy, co-creation, wrap-up | Yes, 13 frames, business lens |
| AI for Tech Leaders | about 75 | 15 files: AI 101 (11), LLMs and agents (9), prompting (7), RAG, tooling, SDLC and more | Yes, AI 101 |
| AI for Project Managers | 41 | One file, `ai_intro_project_managers` | Mixed into the PM story |
| Kids, WithML, Career, Frugal | n/a | Different purpose | Out of scope |

Findings:
1. "What is AI, what is generative AI" is written separately in at least six decks, in different depth and tone.
2. CRAFT exists only in `llm_promptengg_educators` (and copied into the Flipped deck). Other audiences get no
   shared prompting framework.
3. Section structure differs in every deck; only Educators separates a general opening from an audience part.
4. Some decks carry large commented blocks (Tech intro 19, ChatGPT intro 13, institution appendix 26), which
   hides how big they really are.
5. `llm_promptengg_sandwich` (14 frames, the peanut butter sandwich demo) is already shared by three decks. It is
   the only part of the core that is truly reused today.

## 3. Proposed structure

### Part A, non-tech core (used first by AI for Authors)

| File | Frames | Built from |
|---|---|---|
| `ai_core_nontech_whatisai.tex` | about 6 | `ai_educators_content` (Big Picture: AI, ML, DL), `ai_intro_nontech` (What is AI, the AI family, ChatGPT) |
| `ai_core_nontech_howgenai.tex` | about 7 | `ai_educators_content` (language model, why "large"), `chatgpt_intro_short` |
| `ai_core_nontech_limits.tex` | about 4 | `ai_educators_wrapup` (failure modes, data), `ai_intro_nontech` (can AI make mistakes, is my data safe) |
| `ai_core_nontech_prompts.tex` | about 12 | `llm_promptengg_educators` (what is a prompt, same task different prompts, CRAFT in 5 frames); optional demo `llm_promptengg_sandwich` |

Target: about 25 frames, 20 to 25 minutes in a one-hour seminar; the demo is optional.

### Part A, tech core (later)

`ai_core_tech_*` with the same four topics, with more depth on training, embeddings, context windows, and
agents. Built from `ai_intro_tech`, `llm_fromzero` and `ai_intro_tech_leaders_ai101`/`llmagents`/`prompting`.
Not needed for AI for Authors, so it comes later.

### Part B, per audience

One set of files named `ai_for_<x>_*.tex`, with a fixed skeleton so all decks feel alike:

1. Opening: why this matters to this audience (2 to 3 frames).
2. The work in stages, each with CRAFT prompt examples (the bulk).
3. Responsible use specific to the audience (3 to 4 frames).
4. Start-small action plan, references (3 frames).

The stages differ by audience. Authors have a time flow (before, during, after writing); managers might have a
task flow; educators have a classroom flow.

### Content file and drivers

```
seminar_ai_for_<x>_content.tex
  \section{Core}      \input{ai_core_nontech_whatisai} ... \input{ai_core_nontech_prompts}
  \section{<x>}       \input{ai_for_<x>_...}
  \section{Wrap-up}   \input{ai_for_wrapup} \input{ai_refs}
Main_Seminar_AI_For_<X>_Presentation.tex   (about_me, content, thanks)
Main_Seminar_AI_For_<X>_CheatSheet.tex
```

Rules that apply (from AUTHORING.md): no em dashes; no floats in files that feed a CheatSheet; two-column
frames need `%` line endings; render and look at pages; compile one driver at a time and ask before compiling.

## 4. Test case: AI for Authors, Part B

| File | Content | CRAFT examples |
|---|---|---|
| `ai_for_authors_intro` | Why authors are wary; the rule of this seminar: you write, AI assists around the writing | none |
| `ai_for_authors_pre` | Before writing: ideas, brainstorming, audience and positioning, web research, outline, table of contents, title options | brainstorm topics; find and summarise sources; stress-test an outline |
| `ai_for_authors_during` | While writing: the text is written by hand. Allowed: diagram preparation and web searches. Not allowed: AI-written content | diagram request (TikZ or sketch description); fact-finding search prompt |
| `ai_for_authors_post` | After writing: thorough editorial review, consistency and style check, layout review (with LaTeX, compile and check against the author's template), proposal and publisher emails, launch posts | editorial review; layout checklist; proposal email; follow-up email |
| `ai_for_authors_responsible` | Manuscript confidentiality (what not to paste into a chatbot), copyright, publisher AI policies, disclosure, keeping your own voice | spot the problem quick check |
| `ai_for_authors_wrapup` | One-page "Pre, During, Post" summary and this-week action | none |

Target: about 25 frames, with Part A about 25, so about 50 frames for a 75 to 90 minute session (the length is an
open question, section 6). Quick-check frames follow the pattern already used in the Educators deck.

Content guardrails to keep it accurate: describe publisher AI policies in general terms and tell authors to read
each publisher's own submission guidelines; no invented statistics; no claims about named companies.

## 5. Stages and model

| Stage | What | Model |
|---|---|---|
| 1 | Create the non-tech core (4 files) from existing material, wire a new content file, driver pair for AI for Authors | Sonnet |
| 2 | Write Part B for AI for Authors (6 files), compile one driver (after asking), render and check pages, `/upgrade-deck` on it | Sonnet |
| 3 | Author reviews AI for Authors | the author |
| 4 | Create the tech core (4 files) from the Tech and Tech Leaders decks | Sonnet |
| 5 | Retrofit the existing decks onto the cores: Educators, Non-Tech, Tech, Business Leaders, Tech Leaders, Project Managers, Flipped | **Opus** |

**Why Opus for stage 5 only.** Stage 5 means judging, across about 400 live frames, which slides are duplicates,
which carry an audience-specific twist that must stay in Part B, and which commented blocks to revive or delete,
without losing material you have already delivered. That is a judgment-heavy merge with a real cost for mistakes.
Stages 1 to 4 are construction on a clear pattern and fit Sonnet. Stage 5 should also be done one deck at a
time, with a before and after frame count for each.

I will tell you when stage 3 is finished and stage 5 is next; that is the point to switch with `/model`.

## 6. Decisions (author, 2026-11-07) and status

- AI for Authors uses the non-tech core; length about 40 frames (one hour). Built: 16 core frames
  (`ai_core_nontech_whatisai` 4, `_limits` 4, `_prompts` 8) plus 24 Part B frames (`ai_for_authors_intro` 3, `_pre` 4,
  `_during` 5, `_post` 6, `_responsible` 4, `_wrapup` 2). Content file `seminar_ai_for_authors_content.tex`; drivers
  `Main_Seminar_AI_For_Authors_{Presentation,CheatSheet}.tex`. Compiled once: Presentation 45 pages, CheatSheet 4 pages,
  copied to `Publications/Presentations`. Waiting for the author's review (stage 3).
- The tech core and non-tech core are the same size: 16 frames each (4 + 4 + 8). Tech core built 2026-11-07:
  `ai_core_tech_whatisai`, `_limits`, `_prompts` (same CRAFT frames with engineering examples, plus the mechanisms
  behind each limit). Checked once with a throwaway test deck (17 pages, 2-page cheat sheet), then the test driver was
  deleted. No deck includes it yet; it will be exercised by the first tech audience deck or the stage 5 retrofit.
- To trim a deck, comment frames in place; never delete them.
- No short sibling for now; add a `_short` later only if needed.
- Three kinds of file: tech core, non-tech core, and many domain-specific Part B sets.
- The peanut butter sandwich demo (`llm_promptengg_sandwich`) is not in the Authors deck; it stays an optional add-on.
- Section divider added at the start of Part B (frame "AI for Authors"), so the Authors deck is now 41 live frames. Use the
  same divider frame at the start of Part B in every AI-for-X deck.
- Promo slides (optional, include in any seminar): `promo_book_airupe`, `promo_book_ai_sakal`,
  `promo_book_mentalmodels_sakal`, `promo_book_gametheory_sakal` (coming soon), `promo_book_gametheory_apress`, and
  one-slide `promo_paic` (the long `paic.tex` and `airupe.tex` stay for the older decks). `promo_macros.tex` shows a
  framed placeholder until the real cover or QR file exists in `images/` (`Cover_<Book>.jpg`, `QR_<Book>.png`).
  The Authors Presentation driver has them as commented `\input` lines.
- Still open: whether "AI for Authors" should also cover academic papers (the deck says books, articles and papers in one
  place and uses book examples); the review of the draft by the author.

## 7. Risks

- The Educators CRAFT files are written for teachers; the generic core must replace education examples with
  neutral ones and keep education examples in Educators' Part B.
- A shared core changes every deck that includes it, so a later fix to a core file affects all of them. This is
  the point, but it needs recompiles of each deck after any core edit.
- `Main_Course_GenerativeAI_*` never compiled; do not include the cores there.
