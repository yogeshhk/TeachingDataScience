# TODO: CareerInDataScience 90-min (full) / 30-min (short) split

Goal: keep `Main_Seminar_Tech_CareerInDataScience_Presentation/_CheatSheet` (90 min) as
the full deck, and add a new `..._Short_Presentation/_CheatSheet` (30 min) driver pair
that reuses the same underlying topic files wherever possible, differing only by which
frames are commented out. Work through phases in order; tick items off one by one.

## Phase 0 — Rename the misnamed shared intro file
- [ ] Rename `ai_intro_tech_short.tex` -> `ai_intro_tech.tex` (content unchanged; it was
      never actually short, just misnamed)
- [ ] Update `\input{ai_intro_tech_short}` -> `\input{ai_intro_tech}` in:
  - [ ] `course_deeplearning_content.tex`
  - [ ] `seminar_artificialintelligencemachinelearning_content.tex`
  - [ ] `seminar_artificialintelligence_tech_content.tex` (incl. its commented alt-line)
  - [ ] `seminar_machinelearning_content.tex`
  - [ ] `seminar_llm_genai_content.tex` (commented line)
  - [ ] `seminar_careerindatascience_content.tex`
- [x] Smoke-test compile (`texify -cp`) — narrowed scope per user preference: only
      compile CareerInDataScience decks going forward, not unrelated decks touched
      only by the rename. `Main_Seminar_Tech_CareerInDataScience_Presentation.tex`
      compiled clean (124 pages, font-substitution warning only).

## Phase 1 — Create the real `ai_intro_tech_short.tex` (~10 slides) — DONE
- [x] Copy `ai_intro_tech.tex` -> new `ai_intro_tech_short.tex`
- [x] Commented out all frames except 11 orientation slides (kept as active, all
      others preserved as commented for future syncing):
  1. Title/divider — "Introduction to AI" (L2)
  2. "Desire" — core idea intuition (L166)
  3. "Relationship between AI, ML, DL" — Nvidia-sourced `ai1` diagram (L282)
  4. "The Modern AI Hierarchy" — AI/ML/DL/GenAI (L315)
  5. "Traditional vs. Machine Learning?" — `tradml` image (L342)
  6. "Types of Machine Learning" (L349)
  7. "Why Machine Learning?" (L362)
  8. "Evolution Timeline" (L514)
  9. "Everyday usage" (L622)
  10. "Summary" divider (L816)
  11. "When to Use AI vs. Traditional Programming" — closing summary (L823)
- [x] Not yet referenced by any driver — fully isolated until Phase 4 wires it into
      the new short content aggregator, so nothing else was at risk

## Phase 2 — Short siblings for career-specific files — DONE
- [x] `career_ai_roles_short.tex` — 8 active frames: divider, "Data Science Roles"
      overview, "Data Scientist Role", "AI Product Manager", "The AI Engineer",
      "The Modern Data Scientist: How the Role Changed", "AI Career Paths: Quick
      Comparison", "What are Skills Needed?" (rest commented, preserved for sync)
- [x] `career_ai_personas_short.tex` — 5 active frames: divider, USER Persona,
      DEVELOPER Persona, RESEARCHER Persona, "Choosing a Persona" (dropped the
      duplicate DEVELOPER detail frame + Low-Code/No-Code detail frame)
- [x] `career_ai_prep_short.tex` — 5 active frames: divider, "Modern Learning
      Resources (2025+)", "AI Skill Maturity Levels: Where Are You?", "Summary
      Steps", "Your 90-Day AI Transition Roadmap"
- [x] None of the three wired into any driver yet — isolated, same as Phase 1

## Phase 3 — Confirm scope exclusions — DONE
- [x] Background / Challenges / Mid-career excluded entirely from the short version
      (their `\section` + `\input` lines commented out in the new aggregator, same
      sync-by-commenting pattern as the topic files)
- [x] References (`career_refs.tex`, 2 frames) kept in the short deck — small enough
      to not cost meaningful time, preserves attribution

## Phase 4 — New content aggregator + driver pair — DONE
- [x] New `seminar_careerindatascience_short_content.tex` — built by copying the full
      `seminar_careerindatascience_content.tex` structure and commenting out the
      Background/Challenges/Mid-career sections, swapping the rest to `_short` topic
      files: `ai_intro_tech_short`, `career_ai_roles_short`, `career_ai_personas_short`,
      `career_ai_prep_short`; `career_refs` kept active
- [x] New `Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex` (mirrors the
      full driver: same preamble/title style, `\input{airupe}`, `\input{paic}`,
      `\input{thanks}` trailer)
- [x] New `Main_Seminar_Tech_CareerInDataScience_Short_CheatSheet.tex` (mirrors the full
      CheatSheet driver, `multicols{3}` per seminar convention)

## Phase 5 — Compile new short drivers to PDF (pre-upgrade-deck sanity check) — DONE
- [x] Full CheatSheet recompiled clean (16 pages) — confirms Phase 0 rename didn't
      break it either
- [x] `Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex` compiled clean
      (37 pages: 31 content frames + title/outline + airupe/paic/thanks trailer)
- [x] `Main_Seminar_Tech_CareerInDataScience_Short_CheatSheet.tex` compiled clean
      (6 pages)
- [x] No compile errors, only harmless font-substitution warnings

## Phase 6 — Run `/upgrade-deck`
- [x] Ran `/upgrade-deck` on `Main_Seminar_Tech_CareerInDataScience_Presentation.tex`
      (full, 90-min). Package audit: `listings` loaded but no verbatim blocks found
      (Task 1a N/A); no `physics`/`quantikz`, no quantum content (Task 1b/1c N/A).
      Findings applied:
      - Task 2 (redundancy): commented out "Impact Examples" in
        `career_ai_background.tex` (superseded by the more current "AI Takes Jobs,
        AI Creates Jobs"); commented out "Analytics Vidhya Learning Path 2017" in
        `career_ai_prep.tex` (dated, superseded by "Modern Learning Resources
        (2025+)") — this block was already commented in `career_ai_prep_short.tex`,
        so it's now in sync
      - Task 5 (understandability): added an "Intuition" block to "LLM Ops Engineer
        (ML Ops 2.0)" in `career_ai_roles.tex` (jargon-dense for a career-transition
        audience); mirrored the same text into the commented copy of that frame in
        `career_ai_roles_short.tex` to keep the two in sync
      - Task 6 (quiz): added one "Quick Check: Roles & Personas" quiz frame at the
        end of `career_ai_personas.tex` (kept light — one quiz for the whole 90-min
        deck, since the audience is mixed professional/career-changer, not a pure
        classroom course); mirrored the same active frame into
        `career_ai_personas_short.tex`
      - Tasks 1/3/4: no technical errors found; existing section structure judged
        sound; no essential content gaps identified (deck already 2024/2025-current)
      - Recompiled full Presentation + CheatSheet clean after edits
- [x] Ran `/upgrade-deck` on `Main_Seminar_Tech_CareerInDataScience_Short_Presentation.tex`
      (short, 30-min), reviewed as its own standalone artifact (only its 32 active
      content frames). Findings applied:
      - Task 3 (structure): renamed section label `Personas` -> `Roles \& Personas`
        in both `seminar_careerindatascience_content.tex` and
        `seminar_careerindatascience_short_content.tex` (clearer TOC entry, applied
        to both aggregators for consistency)
      - Task 5 (understandability): the short deck's one active Intuition callout
        (LLM Ops) is commented out in the short cut, leaving the image-only Nvidia
        "Relationship between AI, ML, DL" slide unexplained for a first-time
        audience — added an Intuition block there in `ai_intro_tech.tex` (full) and
        mirrored into `ai_intro_tech_short.tex` (both active there)
      - Task 2 (redundancy): reviewed the "Data Science Roles" overview vs. the
        detail frames actually kept in the short cut (only Data Scientist + AI
        Engineer get their own frame, though the overview lists 6) — judged
        acceptable as an intentional "landscape then zoom-in" pattern, not true
        redundancy; also reviewed the intro's tradml comparison vs. the closing
        "When to Use AI vs. Traditional Programming" slide — judged an intentional
        bookend (visual intro + actionable recap), no change
      - Task 6 (quiz): kept to the one quiz already synced from the full-deck pass
        (Roles \& Personas section end) — consistent with the "light application,
        mixed audience" scoping decision
      - Tasks 1/1a/1b/1c/4: no findings beyond the full-deck pass (same package
        audit applies); deck judged appropriately scoped for 30 minutes
      - Recompiled all 4 CareerInDataScience drivers (full + short,
        Presentation + CheatSheet) clean after these edits

## Phase 7 — Recompile after upgrade-deck — DONE
- [x] `texify -cp` all 4 drivers (full Presentation/CheatSheet, short Presentation/
      CheatSheet) — folded into Phase 6's verification after each round of edits
- [x] All 4 confirmed clean: full Presentation 124p, full CheatSheet 16p, short
      Presentation 39p, short CheatSheet 6p

## Phase 8 — Doc update
- [ ] Update `CLAUDE.md` with the rename + the new short/full seminar-pair convention
      (first precedent of this kind in the repo)
