# Local LLM on this machine — analysis and plan

Written 2026-08-13, originally kept at the `GitHub/` root since it applied across repos
(`TeachingDataScience`, `MidcurveNN`, Claude Code itself); moved here 2026-08-31 since its strongest
concrete use case (see section 6) is `TeachingDataScience` teaching demos.

---

## 1. Your hardware, measured

| | |
|---|---|
| GPU | **NVIDIA GeForce MX570 A — 2 GB VRAM**, compute capability 8.6 (Ampere) |
| iGPU | Intel Iris Xe (shares system RAM) |
| RAM | **15.7 GB** |
| CPU | i7-1255U — 2 performance cores + 8 efficient cores |

Two numbers decide everything below.

**2 GB VRAM.** After Windows and the display take their share you have roughly **1.3–1.6 GB
usable**. That is enough for a ~1B model at 4-bit, and nothing more. The MX570 is a thin-and-light
laptop GPU; its memory bandwidth (~96 GB/s) is closer to fast system RAM than to a desktop card.

**15.7 GB RAM.** Leaving ~6 GB for Windows, you have **~9 GB** for a model running on CPU.

The two work as a **tiered budget, not a fallback**. Anything that fits entirely in VRAM runs at GPU
speed, which measurement showed is 3× the CPU. Anything that does not fit spills to CPU and slows
sharply. So the goal is not "get a big model working" — it is **stay under the VRAM line**, because
the cliff at that boundary is steep. Measured free VRAM is ~1.84 GB, so the practical target is a
quantized model of ~1.2–1.4 GB plus its KV cache.

### Measured, not estimated

**Qwen3 1.7B Q4_K_M, full GPU offload (28/28), 4096 context: 60 tok/s.**

That single measurement recalibrated this whole document. The MX570's ~96 GB/s memory bandwidth
puts a hard ceiling near **80 tok/s** for a 1.19 GB model (one full weight read per token), so 60
is roughly 75% of theoretical peak. The GPU is genuinely doing the work — CPU-only on this i7 would
be 15–25 tok/s for the same model — and it is running close to optimally, so there is no tuning
left worth chasing at this size.

Revised expectations, anchored to that real number rather than to guesswork:

| Model size | Fits where | Expected speed |
|---|---|---|
| 0.5–1B | GPU, full offload | 70–100 tok/s |
| **1.5–2B** | **GPU, full offload** | **~60 tok/s (measured)** |
| 3–4B | GPU partial + CPU | 15–30 tok/s |
| 7–8B | Mostly CPU, ~4.5–5 GB | 4–8 tok/s |
| 14B+ | Not practical | — |

The practical consequence: **anything up to ~4B is comfortably interactive on this laptop**, which
is a better position than the hardware specs suggest on paper. The 2 GB VRAM figure looks
disqualifying and mostly isn't, because models in this class are small enough to sit entirely in
VRAM where the bandwidth advantage over CPU is large.

Caveats that still apply: long prompts slow generation, and prompt *processing* (prefill) is a
separate cost that grows with context — a 4000-token prompt takes noticeable time before the first
token appears, even at 60 tok/s output.

---

## 2. The direct answer on Nemotron 3.5 Lightning

**It will not run on this machine.** Not "slowly" — it does not fit.

NVIDIA shipped Nemotron 3.5 Lightning on 2026-08-11: a **30B Mixture-of-Experts with 3B active
parameters**, hybrid Mamba-2 + MoE + attention, 1M-token context, OpenMDW-1.1 licence. It is
distilled from Nemotron 3 Ultra and aimed at the "execution layer" of agent systems — tool calls,
retrieval, validation, formatting, classification.

The "3B active" number is what misleads. **Active parameters govern compute per token, not memory.**
An MoE has to keep *all* experts resident, because the router can select any of them at any token.
So you pay 3B of arithmetic but 30B of RAM:

| Precision | Weights alone |
|---|---|
| BF16 | ~60 GB |
| 8-bit | ~30 GB |
| 4-bit | **~16–17 GB** |

Against 15.7 GB of total system RAM, the 4-bit build does not fit even before the OS, the KV cache,
or your other applications. NVIDIA's own framing is "runs on 1 H100" (80 GB).

What "3B active" *does* buy is speed **for people who can already hold it in memory**. It is a
throughput optimization for datacentre agents, not a way onto small hardware. There is currently no
sub-1 GB version, and there cannot be one at 30B total.

**Verdict: skip it.** Revisit only if you get a machine with 32 GB+ RAM, and even then it will be
CPU-slow.

---

## 3. What to actually run — options

Sizes are 4-bit GGUF approximations. All are open-weight and free.

### Tier A — fits in your 2 GB VRAM (fast, genuinely interactive)

| Model | ~Size | Good at | Notes |
|---|---|---|---|
| **Qwen3 1.7B** | ~1.1 GB | General, instruction following | Best all-round at this size |
| **Llama 3.2 1B** | ~0.8 GB | Simple chat, summarizing | Very fast, limited reasoning |
| **Qwen2.5-Coder 1.5B** | ~1.0 GB | Code completion, small refactors | Surprisingly usable for boilerplate |
| **Moondream2** (~1.8B) | ~1.2 GB | **Vision** — captions, OCR-ish, VQA | Tiny multimodal, purpose-built |
| **SmolVLM 500M / 2.2B** | 0.4 / 1.5 GB | **Vision**, very light | The 500M runs on almost anything |

### Tier B — CPU, still comfortable (the sweet spot for you)

| Model | ~Size | Good at | Notes |
|---|---|---|---|
| **Phi-4-mini (3.8B)** | ~2.5 GB | Reasoning, maths, structured output | Best small reasoner; strong for teaching demos |
| **Gemma 3 4B** | ~3 GB | **Vision** + 140 languages | Best small multimodal all-rounder |
| **Qwen2.5-VL 3B** | ~2.2 GB | **Vision**, documents, charts | Better than Gemma on document/diagram reading |
| **Qwen3 4B** | ~2.6 GB | General + light reasoning | Good default if you don't need vision |

### Tier C — CPU, slow but works (3–6 tok/s)

| Model | ~Size | Good at |
|---|---|---|
| **Qwen2.5-Coder 7B** | ~4.5 GB | The best small coding model (~76% HumanEval) |
| **Qwen3 8B** | ~5 GB | General purpose, noticeably better reasoning |

Tier C is fine for batch work you walk away from. It is *not* fine for interactive chat.

### Recommendation

Install **three** and stop:

1. **Qwen3 1.7B** — your fast default, runs on the GPU.
2. **Phi-4-mini** — when you need it to actually reason.
3. **Qwen2.5-VL 3B** — when the task involves an image, diagram, or scanned page.

---

## 4. Which runtime

| | LM Studio | Ollama | llama.cpp |
|---|---|---|---|
| Interface | GUI + server | CLI + server | CLI |
| Model discovery | Built-in browser | `ollama pull` | Manual GGUF |
| OpenAI-compatible API | Yes | Yes | Yes |
| **Anthropic `/v1/messages`** | Yes (since 0.4.1) | Yes (since Jan 2026) | Yes |
| Windows GPU offload | Easy slider | Automatic | Manual flags |
| Best for | **Starting out, experimenting** | **Scripting, LangChain, always-on** | Squeezing performance |

**Use LM Studio first.** The GPU-offload slider matters a lot on a 2 GB card — you will be
hand-tuning how many layers sit on the GPU versus CPU, and doing that in a GUI while watching
tokens/sec is far easier than guessing at flags.

Add **Ollama** later if you want a background service for LangChain/LangGraph work. The two coexist
fine on different ports.

---

## 5. Can you use it inside Claude Code?

Technically yes. **Practically, no — not on this machine.**

The wiring is real and officially supported: point Claude Code at a local server with
`ANTHROPIC_BASE_URL`, and Ollama, LM Studio and llama.cpp all now speak the Anthropic Messages API
natively, so no translation proxy is needed.

```bash
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_API_KEY=""
```

Three things kill it for you:

1. **Context.** A coding agent needs 32K tokens as a floor, 64K to be comfortable. KV cache at 32K
   on a 7B model costs several GB on top of the weights. You do not have the headroom.
2. **A known Claude Code quirk.** Claude Code attaches an attribution header to every request, which
   invalidates the KV cache on local servers — reportedly making inference **~90% slower**. On
   hardware already at 3–6 tok/s, that is unusable.
3. **Model capability.** Agentic coding needs reliable multi-step tool calling. Models in your size
   range are not dependable at it; you will spend more time correcting than writing.

**Recommendation: keep Claude Code on the cloud models.** You already have Claude Code and Groq —
that is the right tool for agentic coding. Use local models for *bounded, non-agentic* tasks
instead. This is not a limitation worth fighting; it is a sensible division of labour.

Same reasoning applies to OpenCode and similar agentic CLIs.

---

## 6. Where local models genuinely pay off for you

This is the part worth being enthusiastic about. On your hardware the wins are real, just not where
you might expect.

### TeachingDataScience — the strongest case

- **Offline classroom demos.** No API keys for students, no rate limits, no network, no cost, and
  nothing breaks when campus wifi does. Students run the same model you do.
- **Teaching how LLMs actually work.** Quantization, KV cache, temperature, context limits, tokens
  per second — all of these become tangible when the model is on the laptop and you can watch it
  slow down. This is genuinely hard to teach against a cloud API.
- **Reproducible assignments.** A fixed local model at temperature 0 gives every student the same
  output, which cloud endpoints cannot promise across versions.
- **A realistic RAG lab.** Small embedding model + Chroma/FAISS + a 1.7B generator runs fine and
  teaches the whole pipeline end to end.

### MidcurveNN Phase II — partly

Local models are useful for the *plumbing*: prompt-format iteration, parsing and repair logic,
pipeline debugging, smoke-testing `create_brep_csvs.py` output. Run 200 cheap local iterations to
get the harness right, then spend Groq calls on the run that counts.

They are **not** useful for the actual QLoRA fine-tuning. That needs far more VRAM than 2 GB, and
Unsloth's Windows support is poor — which is exactly what you ran into. Fine-tuning stays on Colab,
Kaggle, or a rented GPU.

### General small tasks worth doing locally

Classification, tagging, extracting structured fields, renaming, summarizing a file, first-pass
translation, generating test fixtures. Anything where the input is short, the output is short, and a
wrong answer is cheap.

**The dividing line:** local for high-volume, low-stakes, short-context, privacy-sensitive work.
Groq or Claude for anything long, agentic, or where being wrong costs you real time.

---

## 7. Step by step — do these in order

### Step 1 — Install LM Studio
Download from `lmstudio.ai`, install, open it.

### Step 2 — Pull one small model and confirm it runs
In the model browser, search **Qwen3 1.7B**, pick a **Q4_K_M** GGUF, download (~1.1 GB).
Load it, and in the sidebar set **GPU offload to maximum**. Send one prompt.
**Checkpoint: note the tokens/sec.** If it is above ~20, the GPU is being used.

### Step 3 — Find your GPU offload ceiling
Reload the model, lowering GPU layers if it fails or thrashes. Repeat until stable.
Write the working number down — it will differ per model.

### Step 4 — Add the two other models
**Phi-4-mini** (Q4_K_M, ~2.5 GB) and **Qwen2.5-VL 3B** (~2.2 GB). Test the VL one on a screenshot.
**Checkpoint: you now have fast / reasoning / vision covered.**

### Step 5 — Turn on the local server
LM Studio → Developer/Server tab → Start. Default `http://localhost:1234/v1`.
Verify:
```bash
curl http://localhost:1234/v1/models
```

### Step 6 — Talk to it from Python
```bash
conda activate genai
pip install langchain-openai langgraph
```
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",          # LM Studio ignores it
    model="qwen3-1.7b",            # must match the loaded model id
    temperature=0,
)
print(llm.invoke("Explain a confusion matrix in two sentences.").content)
```
LangChain, LangGraph, LlamaIndex and CrewAI all accept an OpenAI-compatible `base_url`, so the same
three lines unlock the whole ecosystem.
**Checkpoint: this is the moment local LLMs become programmable for you.**

### Step 7 — Build one real thing
Pick a genuinely small task from `TeachingDataScience` — a notebook cell that classifies student
questions by topic, or summarizes a lecture transcript. Ship it end to end.

### Step 8 — Only then, decide about Ollama
If you find yourself wanting it always-on and scripted rather than clicked:
```bash
winget install Ollama.Ollama
ollama pull qwen3:1.7b
```
Its OpenAI-compatible endpoint is `http://localhost:11434/v1` — change one line in step 6.

### Step 9 — Do NOT wire Claude Code to local models
See section 5. Revisit only on very different hardware.

---

## 8. Honest summary

- **Nemotron 3.5 Lightning is out of reach**, and "3B active" does not change that — MoE needs all
  30B resident, ~16 GB at 4-bit, against 15.7 GB total.
- **Your ceiling is roughly a 4B model**, comfortably, with 7B for batch work you don't watch.
- **The GPU is the engine for anything that fits in it** — measured at 60 tok/s on Qwen3 1.7B,
  about 3× CPU-only and ~75% of the card's bandwidth ceiling. An earlier draft of this document
  called the GPU "a minor assist"; that was wrong, and the measurement corrected it. Optimize for
  *fitting inside 1.84 GB of VRAM*, not for CPU threads.
- **Teaching is the strongest use case**, more than coding. Offline, reproducible, and it makes the
  concepts visible in a way a cloud API cannot.
- **Keep agentic coding on Claude Code and Groq.** That division is the correct answer, not a
  compromise forced by the hardware.
- **If you ever want the local option to genuinely open up**, more system RAM (32 GB+) buys far more
  than a slightly better GPU, because on this class of machine you are running on the CPU anyway.

---

## Sources

- [NVIDIA Nemotron 3.5 Lightning — NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-nemotron-3-5-lightning-delivers-fast-accurate-specialized-task-execution-for-long-running-agents/)
- [NVIDIA releases Nemotron 3.5 Lightning — MarkTechPost](https://www.marktechpost.com/2026/08/11/nvidia-ai-releases-nemotron-3-5-lightning-and-nemo-switchyard/)
- [nvidia/nemotron-3.5-lightning — LM Studio](https://lmstudio.ai/models/nvidia/nemotron-3.5-lightning)
- [Best LLMs for Low-VRAM GPUs in 2026 — SiliconFlow](https://www.siliconflow.com/articles/en/best-LLMs-for-low-VRAM-GPUs)
- [Best Small Language Models 2026 — Local AI Master](https://localaimaster.com/blog/small-language-models-guide-2026)
- [Use your LM Studio Models in Claude Code — LM Studio](https://lmstudio.ai/blog/claudecode)
- [Claude Code — Ollama docs](https://docs.ollama.com/integrations/claude-code)
- [Running Claude Code with a Local LLM in 2026](https://www.shawnmayzes.com/ai-engineering/claude-code-local-llm-2026/)
