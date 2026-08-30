# mychatbot — Claude Code Project Intelligence

## Project Overview
A chatbot with two interfaces: CLI and Streamlit, powered by LangChain + Groq LLM.

## Stack
- **Language:** Python 3.10+
- **Framework:** FastAPI (API layer), Streamlit (UI)
- **LLM:** LangChain + Groq (`langchain-groq`)
- **Testing:** pytest with Faker for synthetic data
- **Logging:** Structured logging via project logger — NEVER use `print()` or stdlib `logging` directly
- **Validation:** Pydantic models for all request/response schemas

## Project Structure
```
src/mychatbot/
  chain.py      # LangChain + Groq chain factory
  cli.py        # Rich CLI chatbot with history
  app.py        # Streamlit multi-turn chatbot
  config.py     # Centralized env-based config
tests/          # pytest suite with synthetic Faker data
docs/
  specs/        # Technical specifications
  decisions/    # Architecture Decision Records (ADRs)
.claude/        # Claude Code configuration (committed)
```

## Coding Conventions
- Type hints required on ALL function signatures
- Async functions for all FastAPI endpoints
- Pydantic models for all external I/O
- Structured logging only — see `src/mychatbot/config.py` for logger setup
- No hardcoded secrets — all config via environment variables
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)

## What Claude Should NEVER Do
- Read `.env` or `.env.*` files (handled by deny rules)
- Use `print()` or `logging.basicConfig()` anywhere in source
- Force push to any branch
- Hard reset git history
- Hardcode API keys, tokens, or passwords
- Run `rm -rf` without explicit confirmation

## Architecture Decisions
See `docs/decisions/` for full ADRs. Key choices:
- FastAPI over Django: async-first, lighter weight for agent endpoints
- Groq over OpenAI: faster inference for chatbot latency requirements
- SSE for streaming: agent endpoints stream via Server-Sent Events

## Running the Project
```bash
conda activate mychatbot
pip install -e ".[dev]"

# CLI interface
python -m mychatbot.cli

# Streamlit UI
streamlit run src/mychatbot/app.py

# Tests
pytest tests/ -v
```
