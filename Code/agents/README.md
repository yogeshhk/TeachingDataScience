# Autonomous Agents PoCs

Proof-of-concept examples for agentic patterns, primarily Microsoft AutoGen, plus
general LangChain-agent and prompt-engineering notebooks that don't belong to any one
framework's dedicated folder.

## Setup
```bash
conda env create -f environment.yml
conda activate agents
```

## Contents by Framework

### AutoGen (`autogen_*`, `agents_agentchat_*`, `agents_AutoGen_*`)
Multi-agent conversation examples using Microsoft AutoGen — function calling, group
chat (with and without visualization), math chat, two-player setups, and feedback from
code execution.

### LangChain Agents (`langchain_*`)
- `langchain_camel_agent.py` — CAMEL (role-playing) multi-agent conversation
- `langchain_transformer_agents.py` — HuggingFace Transformers + LangChain agent

### Prompt Engineering (`pe-*`, `pe_*`)
Standalone prompt-engineering notebooks (ChatGPT, Gemini, Mixtral, Code Llama, RAG,
ReAct, PAL, function calling, litellm) — not agent-framework-specific.

### Other
- `agents_A2A-MCP-Tutorial_PavanBelagatii.ipynb` — Agent2Agent/MCP protocol tutorial
- `Manim-Coder-FineTuning-Experiment.ipynb` — unrelated fine-tuning experiment
- `convert-llama-ggml-to-gguf.py`, `download_llama2.py` — model utility scripts

## Note
CrewAI and LangGraph content has been consolidated into their own dedicated top-level
directories (`Code/crewai/`, `Code/langgraph/`) — including the CrewAI-in-LangGraph
hybrid examples, which live in `Code/langgraph/` since LangGraph is the driving
framework there. This folder is now the AutoGen home plus cross-framework/general PoCs.
