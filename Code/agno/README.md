# Agno Agent Examples

Minimal agent scripts demonstrating the [Agno](https://github.com/agno-agi/agno) framework for building AI agents backed by local LLM servers.

## Prerequisites

- **LM Studio** running locally at `http://localhost:1234` (or adjust the base URL in scripts)
- A model downloaded in LM Studio (scripts default to `qwen/qwen3-1.7b`)

## Files

| File | Description |
|------|-------------|
| `reasoning_agent.py` | Agent that streams step-by-step reasoning using `show_full_reasoning=True` |
| `trial_agent.py` | Minimal single-turn agent using a local LM Studio model |
| `web_search_agent.py` | Agent augmented with a web search tool |
| `rag_agent.py` | Stub — placeholder for a future RAG-over-documents example |

## Quick Start

```bash
conda env create -f ../agents/environment.yml   # reuse agents env or install agno manually
pip install agno

python trial_agent.py
```

## Notes

- All scripts use LM Studio as a local OpenAI-compatible inference server — no cloud API key required.
- To switch models, change the `id` parameter in the `LMStudio(model=Model(id="..."))` call.
