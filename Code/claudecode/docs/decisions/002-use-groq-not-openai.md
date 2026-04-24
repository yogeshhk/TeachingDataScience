# ADR 002 — Use Groq, Not OpenAI

**Date:** 2026-03-01  
**Status:** Accepted

## Context
We need a hosted LLM provider for the chatbot's inference layer. The chatbot is latency-sensitive — users expect token streaming to begin within ~500ms of submitting a message.

## Decision
Use **Groq** as the primary LLM provider instead of OpenAI.

## Rationale
- Groq's LPU (Language Processing Unit) hardware delivers significantly lower time-to-first-token (TTFT) compared to GPU-based providers.
- LangChain's `langchain-groq` integration is a drop-in replacement for `langchain-openai` — model switching requires only a config change.
- Free tier available for development and testing via `console.groq.com`.
- Supports `llama-3`, `mixtral`, and `gemma` models — no vendor lock-in on the model side.

## Consequences
- LLM calls go to Groq's API endpoint; outages at Groq affect the entire chatbot.
- Model availability is limited to Groq's hosted selection — custom fine-tuned models not supported without migration.
- `GROQ_API_KEY` must be set in the environment (documented in `.env.example`).

## Alternatives Considered
- **OpenAI:** More model variety but higher TTFT and cost at scale.
- **Anthropic API (Claude):** Excellent quality but higher cost and TTFT for a chatbot use case.
- **Ollama (local):** Zero cost and privacy, but requires GPU hardware and adds ops complexity.
