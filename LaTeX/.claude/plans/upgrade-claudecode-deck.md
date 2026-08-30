# Upgrade Plan: Main_Seminar_AI_ClaudeCode_Presentation
# Session: 2026-06-25

## Status: ALL PHASES COMPLETE

### Files Modified
- `seminar_ai_claudecode_content.tex` — removed outer Demo section; demo file now manages its own sections
- `ai_tools_claudecode_demo.tex` — 8 targeted edits (see below)

### Files Created
- `ai_tools_claudecode_tokenopt.tex` — NEW: 8 Token Optimization frames

---

## Phase Checklist

- [x] Phase 1: Technical fixes to ai_tools_claudecode_demo.tex
  - [x] 1a. Removed garbled `??` emoji before Congratulations line (Step 5: Deploy frame)
        Also fixed lstlisting placement (moved congratulations text before \end{lstlisting})
  - [x] 1b. Fixed subscription text: "Claude Pro subscription" → "Claude Pro/Max subscription or API key"
  - [x] 1c. Verified no \begin{verbatim} blocks in any file (all already using lstlisting)
  - [x] 1d. Commented out duplicate Installation slide in demo file

- [x] Phase 2: Created ai_tools_claudecode_tokenopt.tex (8 frames)
  - Frame 1: Section divider "Token Optimization"
  - Frame 2: Why Token Budgeting Matters
  - Frame 3: Choose the Right Model (Haiku/Sonnet/Opus table + /model demo)
  - Frame 4: Plan Mode as Free Exploration + Ask Before Acting pattern
  - Frame 5: Break Tasks Down + TodoWrite
  - Frame 6: Session Continuity (named sessions + plan file technique)
  - Frame 7: Context Hygiene (/compact, @file:, subagents, skills)
  - Frame 8: Memory + Custom Commands

- [x] Phase 3: Basic/Advanced split (via section markers in demo file)
  - [x] Added \section[Basic]{Basic (1 Hour)} at top of ai_tools_claudecode_demo.tex
  - [x] Changed opening divider title from "Claude Code" to "Basic Demo (1 Hour)"
  - [x] Updated Workshop Overview duration text
  - [x] Updated SDLC table note (Stages 1-4 = Basic, Stages 5-14 = Advanced)
  - [x] Inserted \section[TokenOpt]{Token Optimization} + \input{ai_tools_claudecode_tokenopt} after Session Management frame
  - [x] Inserted \section[Advanced]{Advanced} before Testing section

- [x] Phase 4: Updated seminar_ai_claudecode_content.tex
  - Removed \section[Demo]{Demo} (demo file now owns its section structure)
  - Final section structure:
    1. \section[Intro]{Introduction} → ai_tools_claudecode_intro.tex (unchanged)
    2. [implicit: ai_tools_claudecode_demo.tex defines Basic, TokenOpt, Advanced internally]
    3. \section[CoWork]{CoWork} → ai_tools_claudecowork.tex (unchanged)

- [x] Phase 5: Removed redundant Customization frame in demo file
  - Was a bare bullet-list of .claude/ anatomy, fully covered in the Intro section

- [x] Phase 6: Written this progress tracker file

---

## Next Steps (if resuming)
All planned work is complete. Remaining optional items:
- Compile with: `texify -cp Main_Seminar_AI_ClaudeCode_Presentation.tex` (from LaTeX/)
- Compile CheatSheet: `texify -cp Main_Seminar_AI_ClaudeCode_CheatSheet.tex`
- Review token optimization slides in the PDF
- Consider adding a "Basic Demo" divider/summary frame at the end of the Basic section
  (to signal the transition before Token Optimization begins)
