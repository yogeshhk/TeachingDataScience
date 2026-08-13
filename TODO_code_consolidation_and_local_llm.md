# TODO: Code/ consolidation + local Qwen integration

Working notes for a two-part effort started 2026-08-13. Part 1 (Code/ folder audit) is
partially executed already; the rest below is confirmed-but-not-yet-done. Delete this file
once everything below is complete or explicitly dropped (matches this repo's usual
todo_*.md convention).

## Part 1: Code/ consolidation — remaining work

**Already done (2026-08-13):** deleted 12 stale/redundant folders after one-by-one
confirmation: `activeloop/`, `animesh1012/`, `awesome_llm_apps/`, `deep_rl/`,
`gcp_notebooks/`, `google_generative-ai/`, `nvidia/`, `prodramp/`, `vizuara/`, `llama/`,
`dl_curiousily/` (superseded by `curiosily_ai_bootcamp/`), `latex/` (2008 dead content,
also removed a case-collision risk with root `LaTeX/`). Kept as-is after review:
`dbpedia/`, `meta_bAbi_tasks/`, `ludwig/`, `mahamarathi/`, `orgpedia/`.

**Still to do:**

1. **`Code/agents/` overlap cleanup.** It mixes CrewAI scripts, LangGraph notebooks, and
   AutoGen scripts, duplicating the dedicated `Code/crewai/` and `Code/langgraph/` folders.
   Decision made: keep `agents/` as the AutoGen home, strip the duplicated CrewAI/LangGraph
   content. **Not yet done** — needs actual file-by-file inspection of `agents/`'s 49 files
   to identify which are `crew_*.py`/LangGraph notebooks (remove) vs AutoGen-specific
   (keep); the earlier audit was folder-level, not file-level.
2. **Gitignore hygiene.** Add `node_modules/`, `.pytest_cache/`, `.ruff_cache/`,
   `.benchmarks/` to `Code/.gitignore`; remove the already-tracked
   `Code/opencode/demo/.opencode/node_modules/` tree (784 files) from the working copy.
3. **`Code/llamaindex/data/WikiTableQuestions*`** — remove the ~10,231 files / 160MB
   benchmark dataset from the working copy, add a `.gitignore` entry so it can still be
   downloaded locally without being tracked.
4. **Docs catch-up in `CLAUDE.md`'s Code/ catalog table:**
   - Add `gnn/`'s other 3 subfolders (`gnn-project-deepfindr/`, `molecule-deepfindr/`,
     `odsc2021-sujitpal/`) — currently only `pyg/` is listed.
   - Add `amd/`, `chromeext/`, `pritamMarathi/` — small, recent (2026), just never catalogued.
   - Remove any stale references to the 12 folders deleted above (check none exist first).

## Part 2: Local Qwen (LangChain/Python) integration

**Model confirmed on disk:** `D:\Yogesh\models\lmstudio-community\Qwen3-1.7B-GGUF\Qwen3-1.7B-Q4_K_M.gguf`
(~1.19GB). Everything needed to load it is already installed in the `genai` conda env —
`llama-cpp-python` 0.2.72 + `langchain-community` 0.4.1 (`ChatLlamaCpp`, in-process, no
server, so it satisfies "not via LM Studio"). CPU-only build (no CUDA in the installed
`llama_cpp` binary); GPU is a thin 2GB MX570A anyway, not expected to matter at this model
size.

**Design decided:** not automatic runtime fallback — write both the `ChatGroq` and
`ChatLlamaCpp` instantiations in code, `ChatGroq` commented out, `ChatLlamaCpp` active by
default. Manual toggle, not try/except logic.

**Groq usage sites found across `Code/`** (15 files, for later rollout once the pilot is
validated) — 14 via LangChain/LangGraph's `ChatGroq`, 1 via the raw Groq SDK
(`parsing/llm_parsing_groq.py`, needs bespoke handling, not a drop-in swap). Full list is
in the original planning conversation if needed again.

**To do, in order:**
1. Benchmark actual CPU inference speed of `Qwen3-1.7B-Q4_K_M.gguf` via `ChatLlamaCpp`,
   standalone, on this machine — currently unmeasured.
2. Test `bind_tools()` reliability on a small tool-calling example — the real risk, since a
   1.7B model's function-calling is meaningfully weaker than Groq's hosted models, and some
   Groq-dependent files (`langchain_v1_createagent.py`, `omni-rag/agent.py`) rely on it.
3. Pilot in `Code/langchain/langchain_v1_models.py` — it already has `ChatGroq` commented
   out as an alternative-model example, exact precedent for the pattern wanted here.
4. If the pilot holds up, decide on centralizing into one shared helper vs. repeating the
   pattern across the other 14 Groq call sites.

## Part 3 (TESTED — not currently viable, 2026-08-13): OpenCode + local Qwen

Evaluate whether **OpenCode** (the open-source, model-agnostic Claude Code alternative;
its workshop already exists in this repo, see below) can be pointed at the locally
downloaded Qwen model.

**Confirmed facts (verified 2026-08-13, don't re-derive from memory):**
- The workshop already exists: `Main_Seminar_AI_OpenCode_{Presentation,CheatSheet}.tex` →
  `ai_tools_opencode_intro.tex` (OpenCode concepts/config) + `ai_tools_opencode_demo.tex`
  (a full SDLC workshop walkthrough) + `ai_tools_opencode_old.tex` (looks like a superseded
  earlier version — not yet confirmed what if anything still uses it, check before touching).
  Paired code lives at `Code/opencode/demo/` (pytest/pyproject-based chatbot project) +
  `Code/opencode/opencode.json` (top-level config) + `Code/opencode/decorators_example.py`.
- `ai_tools_opencode_demo.tex` **already has 2 frames** titled "(Optional) Connect OpenCode
  to Local Models via LM Studio" — local-model usage is not a new topic for this deck, it's
  an existing optional section.
- OpenCode is architecturally different from the Python/LangChain case: it's an external
  CLI tool, not a library `Code/` scripts import, so it cannot in-process load a GGUF the
  way `llama-cpp-python` can. It must talk to *some* local server — the "avoid LM Studio"
  constraint that applied to the Python integration doesn't cleanly transfer here; OpenCode's
  own docs list "Local: Ollama" as its offline option, and LM Studio's OpenAI-compatible
  server is the other realistic route. This needs an explicit decision, not an assumption.
- `Code/opencode/opencode.json` is **already configured** to use a local model via LM
  Studio: `"model": "lmstudio/qwen2.5-coder-7b-instruct"`, `baseURL http://localhost:1234/v1`.
  **But that exact model is not present on disk** — only `Qwen3-1.7B-Q4_K_M.gguf` is (see
  Part 2). So the existing config is stale/untested, not a working setup to build on as-is.

**Decision made (2026-08-13):** LM Studio via its `lms` CLI, driven headlessly (not the
GUI) — no new install needed (Ollama would've required one), and `lms` gives genuine
programmatic control (`lms server start`, `lms load`).

**Executed and tested (2026-08-13) — result: NOT currently viable, do not update the
workshop deck to claim this works.**

1. Started the server headlessly: `lms server start` → confirmed reachable at
   `http://localhost:1234/v1/models`, which also confirmed the real model ID:
   `qwen/qwen3-1.7b` (not `qwen2.5-coder-7b-instruct` — that model was never downloaded).
2. Fixed `Code/opencode/opencode.json`: `model` and the `provider.lmstudio.models` key
   both corrected to `qwen/qwen3-1.7b`.
3. First real test — `opencode run "..." -m lmstudio/qwen/qwen3-1.7b` from
   `Code/opencode/` — **failed immediately**: `400 exceed_context_size_error`, OpenCode's
   actual request was 32,111 tokens against the model's default-loaded 4,096-token context.
4. Reloaded the model with a 32K context via `lms load qwen/qwen3-1.7b --context-length
   32768 --gpu off` (confirmed loaded successfully, 1.19 GiB).
5. Retried the same `opencode run` — **failed again**, differently: `SSE read timed out`
   after 2m03s, no response at all.
6. To isolate the cause, tested LM Studio directly with a trivial prompt (bypassing
   OpenCode): `curl .../v1/chat/completions` with a 14-token prompt → responded in **3.6s**.
   So small-prompt inference is fine on this CPU-only setup. The confirmed difference is
   request size: OpenCode's real payload (~32K tokens, presumably `AGENTS.md` + tool/MCP
   schemas + project context) versus a trivial one.
   - Also observed in that direct test: Qwen3 defaults to a "thinking" mode — the response
     had `reasoning_content` full of chain-of-thought and empty `content`, cut off by
     `max_tokens` before any real answer. This weighted the 3.6s response with 20 tokens of
     *thinking*, not answer — a likely compounding factor on top of the 32K-token CPU
     prefill cost, though not independently isolated, this is a reasoned hypothesis, not a
     separately confirmed fact.
   - **Conclusion:** CPU-only prefill of OpenCode's actual ~32K-token request, likely
     compounded by Qwen3's default thinking-token overhead, makes this setup impractically
     slow for OpenCode's real usage (2+ minutes with no response, vs. a live workshop's
     needs). Not a config typo to fix — a genuine capability/hardware limit at this model
     size without GPU acceleration.

**Left running:** LM Studio's API server is still up (`lms server start`), and the model
is still loaded with the 32K context override (persists until LM Studio restarts or
another `lms load`/`lms unload` call) — both harmless to leave, but worth knowing if you
poke at LM Studio for something else in the meantime.

**Not done, only if picked back up later:**
- Try disabling Qwen3's thinking mode (may cut response overhead materially).
- Investigate why OpenCode's request is ~32K tokens by default (AGENTS.md size? MCP tool
  schema count?) — trimming it might make this workable even on CPU.
- `ai_tools_opencode_demo.tex`'s existing "Local Models via LM Studio" frames were **not**
  touched — they don't currently claim a specific verified model/result, so nothing there
  is now inaccurate; there's just nothing new confirmed to add yet either.
- `Code/opencode/opencode.json`'s model reference is fixed (step 2 above) regardless of
  whether the full workflow is usable — at least it no longer points at a model that
  doesn't exist on disk.

## Part 4 (DONE, 2026-08-13): split docs/Interview/src/ out into Code/interview/

`Admin/Interview/` was already renamed to `docs/Interview/` externally (not by this
session). It currently mixes real code and docs under one folder, unlike every other
subject in this repo:
- `docs/Interview/src/` (67 files) — LeetCode/GeeksforGeeks practice solutions (`.py`),
  `1_Introduction.ipynb` (PyTorch tutorial), `pytorch_0{1,2,3}_*.py` (PyTorch basics),
  `data/` (includes `Customer_Churn.csv`), plus `.idea/` (JetBrains IDE config) and
  `desktop.ini` (Windows folder metadata) — the latter two are cruft, not content.
- `docs/Interview/docs/` (5 markdown files + `images/`) — genuine interview-prep notes.

**Executed:** all 65 files from `docs/Interview/src/` moved to new `Code/interview/`
(`.idea/` and `desktop.ini` deleted — verified standard auto-generated PyCharm/Windows
files, nothing custom). Added `Code/interview/environment.yml` (python 3.11, numpy,
pandas, jupyter, torch via pip) and a short `README.md`. `docs/Interview/docs/` flattened
up to `docs/Interview/` directly (5 markdown files + `images/`) now that `src/` is gone —
`docs/Interview/` now holds only interview-prep docs, `Code/interview/` holds only code.
