# AI for Project/Program Managers: refactor status (DONE)

Drivers: Main_Seminar_AI_For_ProjectManagers_Presentation.tex, Main_Seminar_AI_For_ProjectManagers_CheatSheet.tex
Content: seminar_ai_for_projectmanagers_content.tex. Topic: ai_intro_project_managers.tex (used by this deck only; no _short sibling).

## Progress
- [x] A. Read files, Before counted
- [x] B. Plan written
- [x] C. Edits applied
- [x] D. Compiled (both drivers), verified, contact sheets read, no layout problems
- [x] E. Final

## Before
- ai_intro_project_managers.tex: 41 live / 41 raw. Content file: one \input. Core not included.
- Note: the topic file does contain \section commands (Why, Plan, Decision, Demo, Tools, Learn, Refs).

## After
- ai_intro_project_managers.tex: 39 live / 41 raw (2 commented).
- Content file: 4 + 4 core (whatisai, limits) + 8 prompts = 16 core frames, 1 divider frame, 39 Part B frames = 56 live frames.
- Latex-audit inputs: 0 of 1 driver blocked, all \input resolve.

## Plan and actions

| Frame | Action | Reason |
|---|---|---|
| Opening banner "AI for Project and Program Managers / Unpack Curiosity Beyond the Job Title" | Commented | Replaced by the Part B divider "AI for Project Managers" (otherwise two banners in a row) |
| Why This Matters Now divider + disruption, myths, Hinton, orchestration, 2 quick checks | Kept | PM-specific. The myths frame is PM-framed, not generic AI myths |
| Planning, Decision Support, Live Demo, Tools sections | Kept | PM-specific |
| Prompting Patterns for Project Planning | Kept | PM-specific use of prompting, builds on CRAFT |
| Risks and Guardrails Before You Start | Commented | Covered by ai_core_nontech_limits (Predicts; Other Limits; What Not to Paste; First Draft) |
| What To Learn Next, Staying Relevant, References | Kept | PM-specific |
| Thank You | Kept | See flag below |

Commented markers: "% [ai-for-x: commented 2026-11-07 | covered by ...]" above each.

## Factual fixes
None. No clear factual errors found in kept frames.

## Flags (not rewritten)
- Duplicate Thank You: topic file has its own "Thank You" frame and the driver also inputs thanks, paic, airupe after the content. Driver tail unchanged.
- Hinton claim ("2016, radiologists obsolete within five years", "A decade on", "roughly 24--25 distinct tasks" vs "$\sim$24" in the quick check): unsourced task count, inconsistent 24-25 vs 24; the reference line is only "2016 remarks". The "A decade on" wording will date.
- Unsourced numbers: "1--2 hours" vs "under 5 minutes" for a WBS, "under 10 minutes" demo, "80\%" of today's value from a general assistant, "thousands of similar plans".
- Stale risk: tool landscape (ChatGPT, Claude, Gemini, Copilot, Jira/Azure DevOps, Asana/Monday, n8n, Zapier, Make) will age; "AI agents \& automation ... emerging category".
- Core frames say "Rule of Thumb"/limits; PM deck Part B "Quick Check" frames are not renumbered (fine).
- The PDF verify tool lists dash-like glyphs on cheat sheet pages 2, 3, 5 (from existing text with "--" ranges and hyphenated text in core and topic, not newly added); not changed.

## PDFs (scratch)
- C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_projectmanagers\Main_Seminar_AI_For_ProjectManagers_Presentation.pdf (64 pages)
- C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_projectmanagers\Main_Seminar_AI_For_ProjectManagers_CheatSheet.pdf (5 pages)
- No build files left in LaTeX\ for these drivers.

## Could not judge
- Whether the author wants the "Why This Matters Now" myths frame trimmed against core "Other Limits" (kept as PM-framed).
