## AI for All, Non-Tech: refactor status

Drivers: Main_Seminar_AI_For_All_NonTech_Presentation.tex, Main_Seminar_AI_For_All_NonTech_CheatSheet.tex
Content: seminar_ai_for_all_nontech_content.tex
Part B divider: AI in Everyday Life

### Progress
- [x] Step A: Before counts recorded
- [x] Step B: plan written
- [x] Step C: edits applied
- [x] Step D: compiled and verified (all contact sheets read, no problems)
- [x] Step E: final

### Before (live frames)

| Input | Live / raw |
| --- | --- |
| ai_intro_nontech | 43 / 45 |
| ai_fluency | 11 / 11 |
| chatgpt_intro_short | 8 / 30 |
| chatgpt_applications | 11 / 16 |
| llm_promptengg_sandwich | 14 / 14 |
| career_ai_midcareer | 9 / 10 |
| chatgpt_conclusion | 12 / 14 |
| ai_refs | 1 / 1 |
| Total | 109 |

### After (live frames)

| Input | Live / raw |
| --- | --- |
| ai_core_nontech_whatisai | 4 |
| ai_core_nontech_limits | 4 |
| ai_core_nontech_prompts | 8 |
| Part B divider (in content file) | 1 |
| ai_intro_nontech | 37 / 45 |
| ai_fluency | 10 / 11 |
| chatgpt_intro_short | 6 / 30 |
| chatgpt_applications | 11 / 16 |
| llm_promptengg_sandwich | 14 / 14 |
| career_ai_midcareer | 9 / 10 |
| chatgpt_conclusion | 12 / 14 |
| ai_refs | 1 / 1 |
| Total | 117 |

Net +8: 16 core frames and 1 divider added, 9 frames commented out. Content is now core plus Part B.

### Commented frames (marker: ai-for-x: commented 2026-11-07)
ai_intro_nontech.tex (6):
- Understanding Artificial Intelligence title frame (superseded by Part B divider and core)
- AI is Already Part of Your Daily Life divider (duplicates new Part B divider)
- What is Artificial Intelligence? (ai_core_nontech_whatisai)
- The AI Family - Different Types divider (ai_core_nontech_whatisai)
- Understanding the AI Family (ai_core_nontech_whatisai)
- How to Talk to AI (Prompting) (ai_core_nontech_prompts)

ai_fluency.tex (1): Prompt Engineering Concepts (ai_core_nontech_prompts)

chatgpt_intro_short.tex (2): What is a Language Models? and Large Language Models - Comparison (ai_core_nontech_whatisai)

### Factual fixes
None made. No clear factual error in a kept frame that could be fixed without a source.

### Flags: stale or unsourced (not rewritten)
- ai_intro_nontech, Free AI Tools: "Google Bard" (now Gemini), "DALL-E Mini" (now Craiyon), "Microsoft Bing Chat" (now Copilot), "Canva Magic Write". Stale.
- ai_intro_nontech, Inspiring Real Stories: "Increased yields by 30%" for Indian farmers is unsourced. The three stories have no source.
- ai_intro_nontech, Real Example: Email Spam Filter: "10,000 spam / 10,000 real emails" is illustrative, not a source.
- ai_intro_nontech, Start Small and Tips: "ChatGPT (it's free!)" is a dated pricing claim.
- ai_intro_nontech, What to Expect in Next 5-10 Years: speculative; "self-driving taxis common" is a dated claim. Will Robots frame: "decades away" is an unsourced prediction.
- ai_intro_nontech, Voice Assistants: "All this happens in 2 seconds" is unsourced.
- chatgpt_intro_short, What is ChatGPT?: "1m users in 5 days (Insta 2.5 months, Spotify 5m, Facebook 10m, Netflix 3.5yrs)": Spotify and Facebook figures lack their unit (months) and are unsourced as written.
- chatgpt_intro_short, GPTs Training compared to human reading: "GPT-4 on 1.4T tokens" is not officially disclosed by OpenAI; sourced only to a LinkedIn post. The same numbers recur in chatgpt_conclusion (shared, not edited).
- chatgpt_intro_short, GPT3 vs ChatGPT: image shows "175 Billion" parameters for ChatGPT; unverified.
- ai_fluency: Transformer 2017 and AI Fluency source note are fine; "Claude" named as an example LLM is fine. Term list may date (reasoning models, RAG).
- ai_core_nontech_limits and the kept frames overlap in message (hallucination, privacy); kept because the kept frames add audience examples.

### Shared-file overlaps left in place (not edited)
- chatgpt_applications: also in seminar_ai_for_all_tech_content and seminar_llm_applications_content. Part B material, kept.
- llm_promptengg_sandwich (14 frames): also in tech, educators, llm_genai_promptengg, llm_promptengg contents. Overlaps CRAFT (role, few-shot, context, constraints) but is a different worked demo; input kept as the Demo section. Author may want to drop it from this deck.
- career_ai_midcareer: also in seminar_ai_career_full_content. Kept as Preparation.
- chatgpt_conclusion: also in tech and workshop_llm contents. Overlaps limits (Dis-advantages: hallucination) and the token-count slides. Kept.
- ai_refs: shared by many decks. Kept.

### Notes and open questions
- ai_intro_nontech.tex ends with Key Takeaways, Final Thoughts, Your Next Steps, Questions to Discuss; these now come before the Fluency and ChatGPT sections (same order as before this refactor). The deck therefore has its "closing" frames mid-deck. Author may want to reorder.
- ai_intro_nontech.tex, ai_fluency.tex and chatgpt_intro_short.tex were used by this deck only (grep, excluding _retired), so edited in place. ai_intro_nontech has no _short sibling.
- Kept "ChatGPT - You May Have Heard About It" (ai_intro_nontech) though it overlaps chatgpt_intro_short "What is ChatGPT?"; judgment call, not clearly covered by the core.
- Not judged: whether the large Part B (about 100 frames) is too long for a non-tech seminar.

### Output
- Presentation: 122 pages, C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_allnontech\Main_Seminar_AI_For_All_NonTech_Presentation.pdf
- CheatSheet: 11 pages, C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_allnontech\Main_Seminar_AI_For_All_NonTech_CheatSheet.pdf
- latex-audit inputs: 0 of 1 drivers blocked. verify_pdf: PASS for both, no automatic flags; all contact sheets read, no overflow seen. No build artifacts left in LaTeX\.
