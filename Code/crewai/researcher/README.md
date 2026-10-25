# Researcher Crew

A multi-agent [crewAI](https://crewai.com) project. Dependencies are declared in `pyproject.toml`
(Python >=3.10, <3.14; `crewai[tools]`); `crewai install` resolves them.

Put your `OPENAI_API_KEY` in `.env`, then run from this folder:

```bash
crewai run
```

Unmodified, it writes a `report.md` with research on LLMs.

To customise: `src/researcher/config/agents.yaml` (agents), `src/researcher/config/tasks.yaml` (tasks),
`src/researcher/crew.py` (logic and tools), `src/researcher/main.py` (inputs).
