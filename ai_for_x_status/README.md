# AI for X refactor (stage 5): live status

Restart point. If a session ends mid-way, read this file, then the per-deck files below, then
`../AI_FOR_X_PLAN.md`. Each deck has its own status file, so parallel work never edits the same file.

Started 2026-11-07 on Opus, the author having approved: refactor every AI-for-X deck onto the shared cores, in
parallel, one worker per deck.

## Cores

- Non-tech core: 16 frames (`ai_core_nontech_whatisai` 4, `_limits` 4, `_prompts` 8).
- Tech core: 21 frames, expanded mid-way at the author's request so that all the technical basics are in it
  (`ai_core_tech_whatisai` 9: definition, hierarchy, traditional vs ML, types of ML, deep learning, next token,
  transformers, generative AI across media, foundation models; `_limits` 4; `_prompts` 8).

## Rules every worker follows

- Comment frames out, never delete them. Mark each with `% [ai-for-x: commented 2026-11-07 | covered by <core file>]`.
- Never comment frames inside a topic file that another driver also uses; change only this deck's content file.
- Do not edit the core files or another deck's files. Part B starts with a divider frame naming the audience.
- Fix only clear factual errors; list every fix. Flag stale or unsourced items; do not rewrite them.
- No em dashes. No git. Compile only the deck's own two drivers; PDFs go to the scratchpad and are copied to
  `Publications/Presentations` only after the author has seen the report.

## Decks

| Deck | Core | Live frames before / after | PDFs (pages) | State |
|---|---|---|---|---|
| AI for Tech Leaders | tech | 75 / 86 | 91 + 9 | done, PDFs not yet copied |
| AI for All, Tech | tech | 159 / 152 | 157 + 15 | done via deck-only copies (`ai_intro_tech_aiforx`, `llm_fromzero_aiforx`, `chatgpt_intro_aiforx`; originals untouched; keep in step by hand), PDFs not yet copied |
| AI for All, Non-Tech | non-tech | 109 / 103 | 108 + 10 | done (sandwich demo commented), PDFs not yet copied |
| AI for Educators | non-tech | 65 / 55 | 63 + 6 | done (inner divider and sandwich demo commented), PDFs not yet copied |
| AI for Business Leaders | non-tech | 51 / 62 | 67 + 7 | done (inner divider commented), PDFs not yet copied |
| AI for Project Managers | non-tech | 41 / 56 | 64 + 5 | done, PDFs not yet copied |
| AI for Authors | non-tech | n/a / 41 | 46 + 4 | done earlier, PDFs copied |

Scratchpad PDF folders (session-specific): `refactor_<deck>` under the session scratchpad.

## Open decisions for the author

1. All Tech: DECIDED 2026-11-07, deck-only copies (originals untouched).
2. Back-to-back dividers: DECIDED 2026-11-07, inner divider commented in both (Educators "Where AI in Education
   Actually Stands" in ai_educators_content; Business Leaders "Demystifying AI" in ai_intro_biz_leaders_demystifying).
   Rebuilt: Educators 77 pages, Business Leaders 67 pages.
3. Educators Flipped deck: DECIDED 2026-11-07, leave as is. It is the delivered 31 Oct 2026 event deck with its own
   timed run of show, and its CRAFT frames use teacher examples that match its Class 9 Physics running example. It
   intentionally does not use the shared core.
6. Stale content: DECIDED 2026-11-07, do nothing for now. Verified fixes ready if wanted later (web-checked
   2026-11-07): Bard is now Gemini and Bing Chat is now Microsoft Copilot (both renamed 2023 to 2024; DALL-E Mini to
   Craiyon is from memory, unconfirmed); the 2023 US AI Executive Order (EO 14110) was revoked 20 Jan 2025 and
   replaced by EO 14179 on 23 Jan 2025 (NIST, Federal Register); OpenAI has not published GPT-4's parameter count or
   training tokens (GPT-4 Technical Report), so "1.4T tokens" is unsupported; current model versions could not be
   confirmed (trackers disagree), so use vendor families instead of version numbers. Locations: `ai_intro_nontech`
   (Free AI Tools, "ChatGPT it's free"), `ai_intro_biz_leaders_strategy` line 185 (executive order),
   `ai_intro_tech_leaders_llmagents` and `_roi` (model names, GPT-4o prices), `ai_intro_tech_aiforx` (landscape,
   timeline, executive order), `chatgpt_intro_short` and `chatgpt_intro_aiforx` line 206 (token counts). Unsourced
   percentages (2x to 10x, 95 percent, 90 percent and others) are listed in each deck's status file.
4. Sandwich demo: DECIDED 2026-11-07, its `\input` commented in the Educators and Non-Tech content files (the file
   itself is untouched and still used by other decks). Rebuilt: Educators 63 pages, Non-Tech 108 pages.
5. PDFs: DONE 2026-11-07. All 12 (6 decks, Presentation and CheatSheet) copied to `Publications/Presentations` and
   `GDrive/ToPrint/Talks`, checksums verified (24 of 24). This overwrote the older Tech Leaders, Business Leaders and
   Educators PDFs in Presentations; the old-name `Main_Seminar_AI_ProjectManagers_*` and `Main_Seminar_AI_TechLeaders_CheatSheet`
   files there are separate and untouched.

## Stage 5 state

All six decks done. Remaining, optional: the Flipped deck (item 3), and refreshing the stale or unsourced content each
deck's status file lists (model names, prices, dated claims, the US 2023 executive order shown as current).

Not in scope: Kids, WithML, Frugal, Career, Educators Flipped, the course and workshop drivers.
Note: `seminar_ai_for_all_tech_content` is also included by `Main_Workshop_AI_*` (not compiled, mega-deck rule).
