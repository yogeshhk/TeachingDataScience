# Autonomous Agents PoCs

Proof-of-concept examples using multiple agentic frameworks: AutoGen, CrewAI, and LangGraph.

## Setup
```bash
conda env create -f environment.yml
conda activate agents
```

## Contents by Framework

### LangGraph (`langgraph_*`)
- `langgraph_rajibdeb_01_how_to_use_crewai_to_solve_math.py` — math problem solving with CrewAI inside LangGraph
- `langgraph_rajibdeb_03_how_to_persist_shared_state_lg.py` — shared state persistence

### LangChain Agents (`langchain_*`)
- `langchain_camel_agent.py` — CAMEL (role-playing) multi-agent conversation
- `langchain_transformer_agents.py` — HuggingFace Transformers + LangChain agent

### AutoGen (`autogen_*`)
Multi-agent conversation examples using Microsoft AutoGen.

### Specialized
- `autoagent_*.py` / `autoagent_*.ipynb` — AutoAgent framework examples

## Note
CrewAI and LangGraph each have their own dedicated top-level directories (`crewai/`, `langgraph/`) with deeper examples. This folder is for cross-framework comparisons and quick PoCs.
