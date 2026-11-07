# AI for Business Leaders: refactor status

Core: non-tech (whatisai, limits, prompts). Started 2026-11-07. State: DONE (all steps complete).

## Progress
- [x] A. Read drivers and files, Before counts recorded
- [x] B. Plan made
- [x] C. Edits applied
- [x] D. Compiled and verified (verify-pdf no flags; contact sheets read)
- [x] E. Final report

## Before (live frames)
| Input | Live |
|---|---|
| ai_intro_biz_leaders_intro | 4 |
| ai_intro_biz_leaders_demystifying | 13 (14 raw, 1 already commented image variant) |
| ai_intro_biz_leaders_possible | 5 |
| ai_intro_biz_leaders_handson | 6 |
| ai_intro_biz_leaders_strategy | 10 |
| ai_intro_biz_leaders_cocreation | 5 |
| ai_intro_biz_leaders_wrapup | 8 |
| Content file total | 51 |

No topic file is shared (only seminar_ai_for_bizleaders_content.tex inputs them). No _short siblings.

## After (live frames)
Intro 4, core whatisai 4 + limits 4 + prompts 8 = 16, Part B divider 1, demystifying 9, possible 5, handson 6, strategy 10, cocreation 5, wrapup 7. Total 63.
Note: core adds 16 frames and 6 deck frames are commented, so the deck grew from 51 to 63.

## Content file
Welcome and Objectives (intro, program framing, section renamed so the TOC does not show "AI for Business Leaders" twice), AI Basics (whatisai, limits), Prompts with CRAFT, then divider frame "AI for Business Leaders", then Demystify, Possible, HandsOn, Strategy, CoCreate, WrapUp as before.

## Plan and actions (all applied)
| File | Frame | Action | Reason |
|---|---|---|---|
| demystifying | The AI Hierarchy, How the Pieces Fit | commented | whatisai |
| demystifying | Hands-On Exercise: Prompt Engineering Basics (text) | commented | prompts |
| demystifying | Hands-On Exercise: Prompt Engineering Basics (image aiaware3) | commented | prompts |
| demystifying | Prompting Patterns Every Leader Should Know | commented | prompts (CRAFT) |
| wrapup | Limits on AI, Staying Grounded | commented | limits (the kept "AI What It Cannot Do" frame also covers the same ground) |
| all others | kept | business lens, Part B, Copilot hands-on |

Kept on purpose: demystifying's own sub-divider "Demystifying AI" (each section has its own divider), What Is AI definition (junior analyst voice), Four Types, Automation to Intelligence x2, Myths vs Reality, Where AI Is in Your Business, Enterprise vs Consumer, AI What It Cannot Do (causality, explainability, drift beyond core).
Side effect: the commented wrapup frame carried a "Change management" point; the adoption playbook frame covers it.

## Layout fixes (no content change)
Four tables used narrow fixed p{} widths and wrapped badly; widths changed to fractions of \linewidth: demystifying Enterprise vs Consumer, possible Opportunity Map, strategy Build vs Buy and Impact-Effort matrix.

## Factual fixes
None made.

## Flags (stale or unsourced, not rewritten)
- demystifying Myths: "2x to 10x more productive" and "95% of business use cases" unsourced.
- strategy Build vs Buy: "90% of internal use cases solved this way", "10x cheaper and faster" unsourced.
- strategy Regulation: "US Executive Order on AI (2023)" listed as current; that order was rescinded in January 2025 (from my own knowledge, not checked in the repo). Stale.
- strategy Regulation: "India's approach: MEITY guidelines, expect regulation within 2-3 years" is dated.
- possible: success-story figures (Schneider 30-40%, Crompton 35%, Tata Steel 99%+, Google DeepMind 15%) unsourced.
- wrapup Future: "Agentic AI at scale (2025-2026)" and "category leaders of 2027" are time-bound and will age.
- strategy: price and email CTR figures ("USD 2-10 per million words", "CTR by 25%") unsourced and quickly stale.
- intro Agenda frame still lists Sessions 1 to 5; now Core and Prompts come before Session 1 content. Left as is; author to decide whether to add a line.
- Part B opens with a divider "AI for Business Leaders" immediately followed by the "Demystifying AI" divider; the author may want to drop one.

## Shared-file overlaps left in place
None.

## Results
- Presentation PDF: 68 pages, C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_bizleaders\Main_Seminar_AI_For_BizLeaders_Presentation.pdf
- CheatSheet PDF: 7 pages, same folder, Main_Seminar_AI_For_BizLeaders_CheatSheet.pdf
- \input check: 0 of 1 drivers blocked, for both drivers. No build files left in LaTeX\.

## Open questions
- Whether to trim demystifying "AI What It Cannot Do" now that core limits exist (kept, content differs).
