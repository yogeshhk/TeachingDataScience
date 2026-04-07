---
paths:
  - "src/mychatbot/agents/**"
  - ".claude/agents/**"
---
# Agent and Subagent Patterns

These rules apply automatically when working inside agent directories.

## Callback Patterns
- Use callback-driven validation, not prompt-based validation
- All agent callbacks must handle both success and error states
- Callbacks must be typed: `Callable[[AgentResponse], None]`

## Tool Docstrings
- Every tool function must have a docstring explaining:
  1. What the tool does
  2. Required parameters and their types
  3. Return value format
  4. Example usage

## Session State
- Never store sensitive data in session state
- Session state keys must use snake_case
- Always check for key existence before reading from session state

## Streaming (SSE)
- Agent endpoints that may take >2s must stream via Server-Sent Events
- Use the project's SSE helper in `src/mychatbot/streaming.py`
- Always send a `done` event when the stream completes

## Error Handling
- Agents must catch and log all exceptions — never let exceptions propagate to the user silently
- Return structured error responses, not raw exception messages
