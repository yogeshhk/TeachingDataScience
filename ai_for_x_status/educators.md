# AI for Educators: refactor status

Drivers: Main_Seminar_AI_For_Educators_Presentation.tex, Main_Seminar_AI_For_Educators_CheatSheet.tex
Content: seminar_ai_for_educators_content.tex. Core: ai_core_nontech_*. Part B divider: "AI for Educators".

## Progress
- [x] A. Before counts recorded
- [x] B. Plan written
- [ ] C. Edits applied
- [ ] D. Compiled and verified
- [ ] E. Final

## Before (live frames, from latex-audit)
| \input | live / raw | used by |
|---|---|---|
| ai_educators_content | 16 / 16 | this deck only |
| llm_promptengg_educators | 15 / 17 | this deck only |
| llm_promptengg_sandwich | 14 / 14 | SHARED (6 other content files) |
| ai_educators_technical_optional | 3 / 3 | this deck only |
| ai_educators_wrapup | 13 / 13 | this deck only |
| ai_refs | 1 / 1 | SHARED (5 other decks) |
| ai_drives_educational | 3 / 29 | this deck only |
Content-file total: 16+15+14+3+13+1+3 = 65 live frames.

Siblings: ai_educators_content_short (12 live / 16 raw) and llm_promptengg_educators_short (9 / 17) are inputted ONLY by
seminar_ai_for_educators_flipped_content.tex (the Flipped deck, not mine). Rule 4 (mirror) conflicts with rule 5
(do not touch another deck's files); I did NOT edit the _short files. The Flipped deck still carries the generic frames.

## Plan (step B)
| File | Frame | Action | Reason |
|---|---|---|---|
| ai_educators_content | section WhatAI + Big Picture AI/ML/DL | comment | core whatisai frame 1 (identical text) |
| ai_educators_content | section HowAI + divider "How Generative AI Actually Works" | comment | divider only introduced the 3 core-covered frames |
| ai_educators_content | What Is a Language Model? | comment | identical in core whatisai |
| ai_educators_content | Why Are They Called Large? | comment | identical in core whatisai |
| ai_educators_content | State, TaskvJob, Programs sections (all other frames) | keep | education specific, Part B |
| llm_promptengg_educators | divider "Prompt Engineering for Educators" | keep | opens the Hands-On section |
| llm_promptengg_educators | What Is a Prompt? | comment | core prompts frame 1 (identical) |
| llm_promptengg_educators | Same Task, Different Prompts | comment | core prompts frame 2 (notice example replaces photosynthesis) |
| llm_promptengg_educators | Why This Matters for You | keep | education intro to hands-on (flag: unsourced numbers) |
| llm_promptengg_educators | Lesson Plan, Quiz, Levels, Feedback, Prompt Library | keep | education specific |
| llm_promptengg_educators | CRAFT overview, Takeaways | comment | identical in core |
| llm_promptengg_educators | CRAFT Context and Role, Action and Fewshots, Tweaks, All Together | comment | see reasoning |
| llm_promptengg_sandwich | whole file | keep \input | shared, Part B technique demo |
| ai_educators_technical_optional | all | keep | optional depth beyond core; context window not in core |
| ai_educators_wrapup | all | keep | education risks, student data, closing |
| ai_refs, ai_drives_educational | all | keep | shared refs / appendix |

Reasoning on teacher-example CRAFT frames: the four CRAFT detail frames repeat the core frames one for one, same
structure, only the example changes (Grade 7 science vs housing society notice). Keeping both would show the audience
CRAFT twice back to back. The education flavour is already carried by the Part B hands-on frames (lesson plan, quiz,
levels, feedback, prompt library), each of which is a worked CRAFT-style prompt for a teacher task. So they are
commented as duplicates; the Grade 7/8 examples stay on disk in the file. If the author wants one teacher-flavoured
CRAFT slide, un-comment "CRAFT, All Together" only.

## Progress update: ALL STEPS DONE (A-E), compiled and verified, contact sheets read, no layout problems.

## After (live frames, from latex-audit)
| \input | live / raw |
|---|---|
| ai_core_nontech_whatisai | 4 / 4 |
| ai_core_nontech_limits | 4 / 4 |
| ai_core_nontech_prompts | 8 / 8 |
| Part B divider (inline in content file) | 1 / 1 |
| ai_educators_content | 12 / 16 |
| llm_promptengg_educators | 7 / 17 |
| llm_promptengg_sandwich | 14 / 14 |
| ai_educators_technical_optional | 3 / 3 |
| ai_educators_wrapup | 13 / 13 |
| ai_refs | 1 / 1 |
| ai_drives_educational | 3 / 29 |
Content-file total: 70 live (before 65). Part B = 54 live. Core adds 16, 12 generic teacher-deck frames commented
(4 in ai_educators_content, 8 in llm_promptengg_educators). inputs check: 0 of 1 drivers blocked.

## Commented frames (marker: covered by ai_core_nontech_*)
ai_educators_content.tex: \section WhatAI + The Big Picture: AI, ML, and Deep Learning; \section HowAI + divider
"How Generative AI Actually Works"; What Is a Language Model?; Why Are They Called Large? (covered by whatisai).
llm_promptengg_educators.tex: What Is a Prompt?; Same Task, Different Prompts; A Framework for Every Prompt: CRAFT;
CRAFT: Context and Role; CRAFT: Action and Fewshots; CRAFT: Tweaks; CRAFT, All Together; CRAFT: The Takeaways
(covered by prompts). The three "Optional/Trimmed" frames already commented there before are untouched.

## Factual / cross-reference fixes
- ai_educators_wrapup.tex, Four Failure Modes: "covered on the previous slide, always" -> "covered in the AI Basics
  section, always" (the previous slide was never the hallucination slide; hallucination is now in core limits).

## Flags (not rewritten)
- llm_promptengg_educators "Why This Matters for You": time-saved table (Lesson plan 60%, Feedback 75%) is unsourced
  ("typical knowledge worker time studies"), and "30 minutes to 3 minutes" (90%) contradicts the 60% and 75% figures.
- ai_educators_content "What Teachers Are Actually Using It For": "grown fast worldwide ... surveys" has no source.
- Hinton 2016 quote, Jensen Huang "AI kills tasks, not jobs" (2026), radiology demand claim: sourced only through the
  references slide (Fortune 2026, NYT); not verified here. "Teacher AI adoption" and Hinton "Nobel laureate" are time sensitive.
- Student Data frame: FERPA is a US law; deck audience is India (the frame says "or local equivalent", left as is).
- Two reference slides at the end: "References: AI in Education" (wrapup) and the generic "References" (ai_refs, shared,
  cites 2018-era AI overviews). Stale and partly redundant; ai_refs not editable (shared).
- Back-to-back dividers: Part B divider "AI for Educators" is followed at once by the "Where AI in Education Actually
  Stands" divider. Left both (the section dividers inside ai_educators_content kept).
- llm_promptengg_sandwich (14 frames, shared by 6 other decks) left in place after Hands-On; it repeats CRAFT-like
  techniques with a sandwich example. Author may want to comment that \input to shorten.
- Two PDF pages (cheat sheet) report en dash glyphs ("2--3", "10--15" in kept frames); numeric ranges, left.

## Shared-file overlaps left in place
- llm_promptengg_sandwich.tex: also inputted by ai_for_all_nontech, ai_for_all_tech, llm_genai, llm_genai_promptengg,
  llm_promptengg content files. Not edited.
- ai_refs.tex: also used by all_nontech, all_tech, kids, withml. Not edited.
- ai_educators_content_short.tex and llm_promptengg_educators_short.tex (siblings): used only by the Flipped deck, so not
  mirrored (rule 5 over rule 4). The Flipped deck still has the generic frames live. Open question for the author.

## Decision log (CRAFT teacher examples)
Commented as duplicates, see reasoning above. Open question: restore "CRAFT, All Together" (Grade 8 fractions) if one
teacher-flavoured CRAFT slide is wanted.

## Outputs
Scratch folder: C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_educators
- Main_Seminar_AI_For_Educators_Presentation.pdf: 78 pages, 4858428 bytes, verify-pdf PASS, no flags
- Main_Seminar_AI_For_Educators_CheatSheet.pdf: 7 pages, 2326009 bytes, verify-pdf PASS, no flags
No PDF or build files left in LaTeX\ for these drivers.
