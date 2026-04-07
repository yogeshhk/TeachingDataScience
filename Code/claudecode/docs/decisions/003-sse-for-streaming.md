# ADR 003 — Stream Agent Responses via SSE, Not WebSockets

**Date:** 2026-03-01  
**Status:** Accepted

## Context
Agent endpoints can take 3–15 seconds to generate a full response. We need a streaming mechanism so users see tokens as they arrive, not after the full response completes.

## Decision
Use **Server-Sent Events (SSE)** for streaming, not WebSockets or long-polling.

## Rationale
- SSE is unidirectional (server → client) which matches our use case exactly — the agent streams tokens; the client sends new messages via standard HTTP POST.
- SSE works over plain HTTP/1.1; no protocol upgrade handshake required.
- FastAPI's `StreamingResponse` supports SSE natively with `media_type="text/event-stream"`.
- Client-side reconnection is built into the SSE spec — no custom retry logic needed.
- Simpler than WebSockets for a one-way token stream.

## Consequences
- All streaming agent endpoints must use the project SSE helper (`src/mychatbot/streaming.py`).
- Each SSE stream must send a terminal `data: [DONE]\n\n` event when generation completes.
- Claude Code rules enforce SSE conventions automatically (see `.claude/rules/agent-patterns.md`).

## Alternatives Considered
- **WebSockets:** Bidirectional — unnecessary complexity for a one-way token stream.
- **Long-polling:** High server resource usage; poor UX for token-by-token streaming.
- **gRPC streaming:** Adds protobuf dependency and binary protocol complexity; overkill for a web chatbot.
