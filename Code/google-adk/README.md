# Google ADK (Agent Development Kit) Examples

Minimal agent scripts using [Google's ADK](https://google.github.io/adk-docs/) (`adk` package) with Gemini models. Demonstrates single-agent, multi-agent, tool use, structured output, and session management patterns.

## Prerequisites

```bash
pip install google-adk yfinance
export GOOGLE_API_KEY=your_gemini_api_key
```

## Files

| File | Description |
|------|-------------|
| `simple_agent.py` | Minimal "hello world" agent |
| `single_agent.py` | Agent with a single tool (stock price via yfinance) |
| `agent_with_multipletools.py` | Agent with stock price, company info, and analyst recommendation tools |
| `agent_with_structured_output.py` | Forces the agent to return a typed Pydantic object |
| `agent_with_session_management.py` | Persists conversation context using `InMemoryStorage` |
| `agent_with_google_search_grounding.py` | Grounds responses with Google Search |
| `multi_agent.py` | Orchestrates a web-search agent and a finance agent via `MultiAgentOrchestrator` |

## Running

```bash
python single_agent.py
python multi_agent.py
```

## Key Concepts

- `Agent(model, tools, instructions)` — the fundamental building block
- `GeminiModel(model_name="gemini-2.0-flash-exp")` — model selector
- `MultiAgentOrchestrator` — routes tasks to specialised sub-agents
- `agent.run()` for single response; `agent.run_stream()` for token streaming

## Security Note

`my_agent/.env` contains a real `GOOGLE_API_KEY` on disk — rotate it at [Google Cloud Console](https://console.cloud.google.com/) if the repo is ever shared.
