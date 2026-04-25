---
paths:
  - "**/*.py"
---
# Python Code Style

These rules apply automatically to all Python files in this project.

## Formatting
- Follow PEP 8 strictly
- Maximum line length: 88 characters (Black-compatible)
- Use trailing commas in multi-line structures

## Type Hints
- Type hints are **required** on ALL function signatures — parameters and return types
- Use `X | Y` union syntax (Python 3.10+), not `Optional[X]` or `Union[X, Y]`
- Prefer specific types over `Any`

## Logging
- Use the project logger from `src/mychatbot/config.py` — NEVER use `print()` or `logging.basicConfig()`
- Log at appropriate levels: `DEBUG` for trace, `INFO` for events, `WARNING` for degraded state, `ERROR` for failures
- Never log sensitive data: tokens, passwords, API keys, PII

## Validation
- Pydantic models for ALL request/response schemas
- Validate external inputs at the boundary — never trust raw user input

## Async
- All FastAPI endpoint handlers must be `async def`
- Use `asyncio.gather()` for concurrent I/O, not sequential awaits in a loop

## Imports
- Standard library first, then third-party, then local — separated by blank lines
- No wildcard imports (`from module import *`)
