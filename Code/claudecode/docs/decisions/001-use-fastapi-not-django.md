# ADR 001 — Use FastAPI, Not Django

**Date:** 2026-03-01  
**Status:** Accepted

## Context
We need a Python web framework to serve the chatbot's API endpoints and stream agent responses to clients. The primary requirements are async-first I/O, low latency for streaming (SSE), and lightweight deployment.

## Decision
Use **FastAPI** instead of Django or Flask.

## Rationale
- FastAPI is built on `asyncio` and `Starlette` — async is the default, not an afterthought.
- Native support for Server-Sent Events (SSE) via `StreamingResponse` without third-party packages.
- Automatic OpenAPI/Swagger docs from Pydantic models — reduces documentation overhead.
- Significantly lighter than Django for a service that has no ORM, admin panel, or template rendering needs.
- Pydantic is already our validation library — FastAPI integrates it natively at the route level.

## Consequences
- All endpoint handlers must be `async def` (enforced by `.claude/rules/code-style.md`).
- No Django ORM — database access (if added later) will use SQLAlchemy async or raw asyncpg.
- Team members unfamiliar with async Python will need onboarding.

## Alternatives Considered
- **Django REST Framework:** Mature but synchronous by default; SSE support requires `django-sse` or channels overhead.
- **Flask:** Simpler but no async support without `quart`; less ergonomic for Pydantic integration.
