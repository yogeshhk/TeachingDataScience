# Seminar AI Frugal: slide outline (draft for review)

Driver: `Main_Seminar_AI_Frugal_{Presentation,CheatSheet}.tex`, aggregator `seminar_ai_frugal_content.tex`,
topic files `ai_frugal_*.tex`. About 39 content slides. Sources named S1 to S22 are in
`FrugalAI/docs/sources.md`; N1 to N6 are new and are listed at the end. Nothing here is written as a `.tex`
frame yet.

Rules for every slide: one source per factual claim, an original diagram or chart rather than a copied
photo, no number that was not fetched or run, and "to verify" shown on the slide if it cannot be sourced.

## 1. Why frugal AI (6)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 1 | The Default Answer: Bigger | Bigger models, cloud APIs, per-token billing as the usual path | framing, no number |
| 2 | What One Prompt Costs | Median Gemini Apps text prompt 0.24 Wh (operator's own measurement); a benchmark of 30 models estimates over 29 Wh for a long prompt on the most energy-intensive ones (estimates from public API data) | S8, S10 |
| 3 | Data Centres in Context | About 415 TWh in 2024 (1.5% of world electricity), about 945 TWh by 2030 (IEA projection) | S5 |
| 4 | What Is Frugal AI? | "Maximizes environmental and societal impact with minimal resources" | S1 |
| 5 | Three Levers | Smaller model, closer hardware, fewer tokens | docs/token_frugality.md |
| 6 | Task-Specific Is Cheaper | Multi-purpose generative models cost far more per 1,000 inferences than task-specific systems in one study (no number quoted) | S9 |

## 2. Why this matters for India (7)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 7 | 22 Languages, One Model? | The 22 scheduled languages, drawn as a list | N1 |
| 8 | Languages Cost More Tokens | A question in English costs about one fifth of the same question in an Indian language, per Sarvam; plus a token-count chart from our own script (optional, only if a tokenizer is available offline) | S3, own run |
| 9 | Connectivity and Devices | Authors' figure of 800 million smartphone users, attributed; why offline matters | S3 |
| 10 | Data Stays With the Community | Soliga text-to-speech: five hours of voice data, Raspberry Pi, offline, community ownership, word error rate "slightly high" | S2 |
| 11 | Health Workers: ASHA | Role of an ASHA per the National Health Mission page; decision support only; no headcount quoted (the page gives none) | N2, docs/safety.md |
| 12 | Indian Efforts | AI4Bharat lightweight systems for low-end phones; Sarvam; Krutrim covering 22 languages | S3 |
| 13 | Heritage on a Laptop: Modi Script | MoScNet: student about 163x smaller than its teacher, 2,043 image dataset | S4 |

## 2b. Why India does not need to race for the frontier (3, added after review)
Framing rule for these slides: the panel remarks the owner pasted are opinions with unnamed speakers, so
each is shown as "a panel view" or left out. Figures come only from the sources named here.
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 7b | The Frontier Race Is Expensive | Frontier capability earns its cost in a narrow band (advanced science, long-horizon reasoning, cyber) and is "frequently the wrong tool" for most public-service problems; owning the full stack is prohibitively expensive for nearly every country; a panel view: being a second or third mover is acceptable; the jobs context, about 7.85 million non-farm jobs needed a year to 2030 (official estimate, assumptions disputed) | S23, S27, panel (opinion) |
| 7c | Frontier-Minus-N: A Decision Rule | Definition ("deliberately built several generations behind the absolute frontier"); the seven questions (task, stakes, data, cost tolerance, auditability, connectivity, pace of change); minister: "smaller, efficient models", India building 12 models of 50 to 120B; PSA white paper: small models as "complementary components"; the "nearly 95 per cent" and the panel's "80%" are shown as unmeasured, differing claims | S23, S24, S25 |
| 7d | Where Small Pays: Eye Screening | Diabetic retinopathy model with Aravind Eye Hospital as foundational partner, more than 600,000 screenings, aim of 6 million over 10 years at no cost to patients; an imaging model, so frugal vision, not an LLM | S26 |

## 3. How small models work (6)
Slides 6, 20 and 33 of the first draft are cut or merged to keep the deck near 40: the task-specific cost
point folds into slide 5, state space models are dropped, and the energy-per-task plan folds into slide 32.
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 14 | What Is a Small Model? | About 0.5B to 4B parameters here; our two test models | docs/concepts.md |
| 15 | Quantisation: Fewer Bits | Chart: Gemma 2 2B file sizes, F32 10.5 GB, Q8_0 2.78 GB, Q4_K_M 1.71 GB (about 6x smaller than F32) | N3 |
| 16 | A 2B Model vs an Arduino | The 1.71 GB file against 256 KB of RAM: thousands of times larger; why microcontrollers run TinyML, not LLMs | N3, N6 |
| 17 | Distillation | Teacher and student, MoScNet as the example | S4 |
| 18 | Memory: The KV Cache | Per-request memory that grows with context | S16 |
| 19 | Mixture of Experts | Mixtral: 8 experts, 2 per token, 47B total and 13B active parameters | S18 |
| 20 | Beyond Attention | State space models and linear scaling (Mamba), authors' claim | S17 |
| 21 | Frozen Encoder Plus Small Head | V-JEPA 2 attentive probe, LeJEPA linear probe, DINOv2 | S19, S20, S21, S22 |

## 4. Where it runs (5)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 22 | The Platform Map | TikZ diagram of the five platforms, by memory and power | own diagram |
| 23 | Old Laptop, CPU Only | AIOOLL on an i3-8130U, 8 GB (about 8 tok/s for Gemma2 2B, as that repo reports); our harness: 53.7 tok/s median for Qwen3 1.7B on a 16 GB PC; Gemma 2 2B number measured after download | AIOOLL README, FrugalAI results |
| 24 | Raspberry Pi 5 | 2.4 GHz quad-core Cortex-A76, 1 to 16 GB RAM, 5 V 5 A supply, active cooling advised; not yet measured by us | N4 |
| 25 | Phones | On-device inference; pointer to the mobile LLM list; our M4 roadmap item | related lists in AWESOME.md |
| 26 | Microcontrollers: TinyML | Arduino Nano 33 BLE Sense: nRF52840, 64 MHz Cortex-M4F, 256 KB RAM, 1 MB flash, microphone and IMU; keyword spotting and sensor tasks | N5, N6 |

## 5. Efforts you can reproduce (4)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 27 | Offline Wiki on an Old Laptop | BM25 plus a small model; 10 of 10 sample questions found at rank 1; retrieval under 1 ms | recipes/offline-wiki-rag |
| 28 | AIOOLL | Six modules of AI on an old Linux laptop | AIOOLL README |
| 29 | A Workflow with a Small Model | LangGraph workflow node on Gemma 2 2B, one model variable; real numbers after download | Code/langgraph |
| 30 | The Awesome List | AWESOME.md sections and entry criteria | AWESOME.md |

## 6. How to start (4)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 31 | Measure First | Decode tok/s, first-token time, peak RAM; the harness command | benchmarks/METHODOLOGY.md |
| 32 | Token-Frugal Practices | Lean context, structured prompts, step budgets, small specialised models; no unverified percentages | docs/token_frugality.md |
| 33 | Energy per Task: A Plan | Power meter first; CodeCarbon and its limits | S12, S14 |
| 34 | A Ten-Minute Start | Download Gemma 2 2B Q4_K_M in LM Studio, run the benchmark | N3 |

## 7. Limits (3)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 35 | Where Small Models Fall Short | Thinking tokens, weak tool calling, no system prompt on Gemma 2; our early-evidence results | own runs |
| 36 | Health and Safety | Decision support only, refusal, review, legal check pending | docs/safety.md |
| 37 | Free Is Not Always Open | Apache-2.0 (Qwen3) versus Gemma terms of use | N3, Qwen3 model card |

## 8. Join in (2)
| # | Slide | Content | Source |
|---|-------|---------|--------|
| 38 | Join In | FrugalAI repo, AWESOME.md entry, internships: "add one result from your device" | repo |
| 39 | Sources | Numbered list of every source above | sources.md |

## Charts and pictures (all original)
Gemma 2 2B file-size bars; energy per prompt (labelled measured versus estimated); a platform map; a
size-versus-memory chart; a workflow diagram; a token-count bar chart if a tokenizer can run offline.

## Sources added after review
S23 (Frugal AI Hub report "Minds, Models & Markets", July 2026), S24 (All India Radio, minister's Davos
interview, 21 January 2026), S25 (Storyboard18 on the PSA white paper, March 2026), S26 (Google blog on
diabetic retinopathy screening, October 2024), S27 (Down To Earth on the Economic Survey 2023-24). All are in
`FrugalAI/docs/sources.md`. Fixed on the way: "frontier-minus-N" is the Hub report's own term, so my earlier
"not confirmed" note was wrong and has been corrected.

## New sources to add to the deck's source list
- N1: Eighth Schedule to the Constitution of India (22 languages).
- N2: National Health Mission, ASHA page.
- N3: lmstudio-community/gemma-2-2b-it-GGUF on Hugging Face (file sizes, prompt format) and Google's Gemma 2
  2B model card (terms of use).
- N4: Raspberry Pi 5 product page.
- N5: Arduino Nano 33 BLE Sense documentation (microcontroller and sensors).
- N6: Nordic nRF52840 page (clock, flash, RAM).

## Open before writing
- Gemma 2 2B Q4_K_M downloaded in LM Studio, for slides 23, 29 and 34.
- Decide whether slide 8's token-count chart is worth a script run.
- India figures not found in a primary source are left out: the ASHA headcount, and any language-speaker
  counts.
