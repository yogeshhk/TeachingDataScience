## AI for All, Tech: refactor status

Updated: 2026-11-07. Step reached: DONE (edited, compiled, verified). See Final at the end.

## Before (content file seminar_ai_for_all_tech_content.tex)

| Input | Live frames | Used by other files? |
|---|---|---|
| ai_intro_tech | 45 | yes (7 other content files) |
| ai_healthcare | 5 | no (this deck only; also reaches workshop_ai via this content file) |
| ai_impact | 15 | yes (seminar_ai_for_withml_content) |
| llm_fromzero | 31 | yes (workshop_llm_content) |
| chatgpt_intro | 17 | yes (workshop_llm_content); chatgpt_intro_short exists |
| chatgpt_applications | 11 | yes (nontech, llm_applications) |
| chatgpt_applications_imi | 8 | yes (llm_applications) |
| llm_promptengg_sandwich | 14 | yes (5 other content files) |
| chatgpt_conclusion | 12 | yes (nontech, workshop_llm) |
| ai_refs | 1 | yes (many) |

Content-file total before: 159 live frames.

Special note: seminar_ai_for_all_tech_content.tex is also \input by workshop_ai_content.tex (Main_Workshop_AI_*). This change flows into that workshop. The workshop driver was NOT compiled.

## Plan

Nine of the ten topic files are shared with other decks, so under rule 3 none of them may be edited and none can be commented in-file. Only ai_healthcare is private, and it holds Part B material (healthcare application), so it stays as is. Whole-file \input comment-outs are allowed only when a whole file is covered by the core; no file is wholly covered, each holds deeper or audience material.

| File | Frame(s) | Action | Reason |
|---|---|---|---|
| ai_intro_tech | AI/ML/DL hierarchy, Traditional vs ML, intelligence, etc. | Keep input (after divider); overlap noted | Shared file. Mixed: overlaps core hierarchy frames but also history, timeline, bias, regulation, AGI, action plan |
| ai_healthcare | all 5 | Keep | Domain application, Part B |
| ai_impact | all 15 | Keep | Industry impact, Part B |
| llm_fromzero | AI-ML-DL frames, Transformers, GPT | Keep input; overlap noted | Shared; deeper technical material (word vectors, seq2seq, transformers) the core lacks |
| chatgpt_intro | What is a language model, training, API usage | Keep input; overlap noted | Shared; training and API detail beyond core |
| chatgpt_applications, _imi | all | Keep | Applications and demos, Part B |
| llm_promptengg_sandwich | Basic Prompt, Roles, Constraints, Few-Shot | Keep input; overlap with core CRAFT noted | Shared; also holds chain-of-thought, self-consistency, reflexion that the core lacks |
| chatgpt_conclusion, ai_refs | all | Keep | Wrap-up and references |

Result: no frame commented out. Content file restructured: Core + Prompts sections from the tech core, then a Part B divider "AI in Practice", then the existing sections in original order.

Open question for the author: the overlap between the new core (16 frames) and ai_intro_tech / llm_fromzero / chatgpt_intro / llm_promptengg_sandwich is large. Trimming it needs the shared files edited (or per-deck copies). Not done here.

## Coordinator change received mid-task

The tech core ai_core_tech_whatisai.tex is being expanded (Types of ML, Deep Learning and neural networks, How Transformers Work, Generative AI for text/images/code). Instruction: treat deck frames on those topics as covered and comment them. At my compile time the core still had 4 live frames (not yet expanded), so my PDFs show the 16-frame core. All deck frames on those topics live in shared topic files (ai_intro_tech: Types of Machine Learning, Why Machine/Deep Learning, Multimodality; llm_fromzero: ML vs DL, Use Deep Learning When, Seq2Seq, Transformers x2, GPT; chatgpt_intro: language model, GPT training). Rule 3 forbids editing them, and no whole file is covered, so none were commented. Follow-up for the author: when the expanded core is final, either approve per-deck copies or edit the shared files (impacting the other decks listed above) to trim these.

## Final

After: 16 core (4 + 4 + 8) + 1 divider + 159 Part B = 176 live frames (content file plus inputs). Commented frames: none. Factual fixes: none.
PDF page counts: Presentation 181 pages (176 + title, outline, 2 about_me, thanks), CheatSheet 17 pages. Both verified (automatic PASS, all contact sheets read, no overflow seen). Scratch:
- C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_alltech\Main_Seminar_AI_For_All_Tech_Presentation.pdf
- C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_alltech\Main_Seminar_AI_For_All_Tech_CheatSheet.pdf
-Mode inputs: 0 of 1 driver blocked. Workshop driver not compiled.

Shared-file overlaps left in place (kept inputs): ai_intro_tech, ai_impact, llm_fromzero, chatgpt_intro, chatgpt_applications, chatgpt_applications_imi, llm_promptengg_sandwich, chatgpt_conclusion, ai_refs.

Flags (stale or unsourced, not rewritten): ai_intro_tech has dated model names and claims (GPT-4o, o3, Claude 3.5, Gemini 2.0, Llama 3.3, "2024" success stories, "AI literacy non-negotiable by 2027", "API costs drop 10x every 12-18 months", GPT-3 552 tons CO2, AGI timelines); chatgpt_intro uses GPT-3/text-davinci and the legacy openai.Completion/Edit API; chatgpt_conclusion "Will ChatGPT kill jobs"/hype slides are 2022-23; ai_impact market-chart images are unsourced on the slide text. Duplicate slides: "After all this - The Truth" (Box quote) appears three times (ai_intro_tech, ai_impact end, llm_fromzero start); AI-ML-DL hierarchy appears in core, ai_intro_tech (x2) and llm_fromzero (x2). "Applications: Mechanical Engineering" section is odd for a general technical audience (open question). Verify tool notes dash-like glyphs on several pages (from existing shared content, e.g. 2022--2025 range); not touched.

## Round 2: deck-only copies (author decision, 2026-11-07)

Final tech core now: ai_core_tech_whatisai 9 frames, limits 4, prompts 8 = 21. Superseded: the earlier "no frame commented" result and the "Coordinator change" note above (frames now commented in deck-only copies).

Step 1 done: copies created byte for byte (originals untouched, sizes unchanged 30938 / 14667 / 17633 bytes) with a header comment line added: ai_intro_tech_aiforx.tex, llm_fromzero_aiforx.tex, chatgpt_intro_aiforx.tex.
Step 2 done: frames commented in the copies only (table below).
Step 3 done: content file now \input the three copies; originals' \input lines commented with marker. Other shared inputs unchanged.
Note: workshop_ai_content.tex inputs seminar_ai_for_all_tech_content, so the workshop also picks up the copies and the new core. Workshop driver NOT compiled.
Copies must be kept in step with the originals by hand.

Per-file live/raw: ai_intro_tech_aiforx 31 live / 64 raw (original 45/64); llm_fromzero_aiforx 17 / 33 (original 31/33); chatgpt_intro_aiforx 16 / 30 (original 17/30).
Deck live frames: before 159 deck frames (Part B only, old core absent). After: 21 core + 1 divider + 130 Part B = 152 (Part B: 31 + 5 + 15 + 17 + 16 + 11 + 8 + 14 + 12 + 1 = 130).
latex-audit -Mode inputs: 0 of 1 driver blocked.

### Decisions (table: file, frame, action, reason)

| File | Frame | Action | Reason |
|---|---|---|---|
| ai_intro_tech_aiforx | Introduction to AI (divider) | commented | covered by whatisai; redundant after the Part B divider |
| ai_intro_tech_aiforx | Data Science, AI is critical... (divider) | commented | intro framing covered by whatisai |
| ai_intro_tech_aiforx | What are Data Sciences? (divider) | commented | covered by whatisai |
| ai_intro_tech_aiforx | Data Science | commented | covered by whatisai |
| ai_intro_tech_aiforx | What is AI? (divider) | commented | covered by whatisai |
| ai_intro_tech_aiforx | What is Artificial Intelligence (AI)? (author definition) | commented | covered by What Is AI working definition |
| ai_intro_tech_aiforx | Relationship between AI, ML, DL | commented | covered by Hierarchy frame |
| ai_intro_tech_aiforx | The Modern AI Hierarchy | commented | covered by Hierarchy frame |
| ai_intro_tech_aiforx | Traditional vs. Machine Learning? | commented | covered by Traditional Programming vs ML |
| ai_intro_tech_aiforx | Types of Machine Learning | commented | covered by Types of ML |
| ai_intro_tech_aiforx | The Foundation Model Revolution (2022--2025) | commented | covered by Foundation Models and How to Adapt Them |
| ai_intro_tech_aiforx | Multimodality: Beyond Text | commented | covered by Generative AI: Text, Code, Images, Audio |
| ai_intro_tech_aiforx | After all this - The Truth (divider) and box quote | commented | duplicate; one live copy kept at the end of ai_impact (shared, uneditable) |
| ai_intro_tech_aiforx | Core idea, Desire, Functions x4, Chess, Intelligence (intuition) | kept | author voice, intuition the core lacks; copies in llm_fromzero_aiforx commented instead |
| ai_intro_tech_aiforx | Intelligence: the differentiation, Why Machine/Deep Learning?, Why now?, Evolution Timeline | kept | not in core (timeline is stale, flagged) |
| ai_intro_tech_aiforx | Model Landscape, Agents, Success Stories, Everyday usage, threat, fear, Mostly, Limits on AI, Bias, Environmental cost, Regulation, Jobs, AGI, Summary divider, When to use AI, Action plan | kept | not in core; Limits on AI kept as author voice (general, not hallucination-specific); landscape and several others stale or unsourced, flagged |
| llm_fromzero_aiforx | What is the Core Idea? divider, core idea, Desire, Functions, Chess, Intelligence | commented | duplicates of the frames kept in ai_intro_tech_aiforx |
| llm_fromzero_aiforx | Relationship between AI, ML, DL; What is AI-ML-DL?; Introduction to Machine Learning divider | commented | covered by whatisai |
| llm_fromzero_aiforx | What is Machine Learning? | commented | covered by whatisai |
| llm_fromzero_aiforx | Traditional vs. Machine Learning? | commented | covered by whatisai |
| llm_fromzero_aiforx | Why Machine Learning? | commented | duplicate of Why Machine/Deep Learning? in ai_intro_tech_aiforx |
| llm_fromzero_aiforx | ML vs DL: What's the difference? | commented | covered by Deep Learning frame |
| llm_fromzero_aiforx | Transformers (2nd, positional encoding and self-attention) | commented | covered by How Transformers Work: Attention |
| llm_fromzero_aiforx | From Zero to GPT, Houston, Whats the Problem, AI, The Problem (hype framing) | kept | author voice on hype |
| llm_fromzero_aiforx | How do we learn?, Model entities, Use Deep Learning When, What is Deep NLP, Typical ML Classification, Context, Word vectors, Seq2Seq, Transformers (1st, encoder/decoder), GPT divider, GPT, GPT-3 in-context learning | kept | deeper material or history the core lacks |
| chatgpt_intro_aiforx | What is a Language Models? | commented | covered by Language Models: Predict the Next Token |
| chatgpt_intro_aiforx | all other frames | kept | ChatGPT-specific material, history, training steps, API demos (stale, flagged) |

Step 4 (compile and verify) pending below.

### Round 2 compile and verify (done)

Both drivers compiled with texify; PDFs moved to the scratch folder (replacing the old ones); no build files left in LaTeX\. verify_pdf: automatic PASS, no flags, every contact sheet read, no overflow, wrapping or cut-off seen.
- Presentation: 157 pages (152 live + title, outline, 2 about_me, thanks), 15.6 MB: C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_alltech\Main_Seminar_AI_For_All_Tech_Presentation.pdf
- CheatSheet: 15 pages: C:\Users\yoges\AppData\Local\Temp\claude\da00bdce62064c21\1ee1a408-8282-46d8-852d-a9745eefaebe\scratchpad\refactor_alltech\Main_Seminar_AI_For_All_Tech_CheatSheet.pdf
Status: DONE.
Open questions for the author: ai_intro_tech_aiforx keeps both "Limits on Artificial Intelligence" (generic) and the core limits; the "Applications: Mechanical Engineering" section; stale flags above still apply (model landscape, timeline, 2024 stories, "by 2027", GPT-3 era API demos in chatgpt_intro_aiforx).
