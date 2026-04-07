# LangGraph Examples

LangGraph code samples covering the core concepts of stateful, graph-based agent orchestration.

## Setup
```bash
conda env create -f environment.yml
conda activate langgraph
cp .env.example .env   # if present, add your API keys
```

## Contents

### Root-level files (CodeBasics series)
Step-by-step progression from simple graphs to production agent patterns:

| File | Concept |
|------|---------|
| `codebasics_1_simple_graph.ipynb` | Graph definition, nodes, edges |
| `codebasics_2_graph_with_condition.ipynb` | Conditional routing |
| `codebasics_3_chatbot.ipynb` | Stateful chatbot with memory |
| `codebasics_4_tool_call.ipynb` | Integrating tools |
| `codebasics_5_tool_call_agent.ipynb` | Full ReAct agent |
| `codebasics_6_memory.ipynb` | Persistent memory across turns |
| `codebasics_7_langsmith_tracing.ipynb` | Observability with LangSmith |
| `codebasics_8_HITL.py` | Human-in-the-loop interrupts |
| `jamesbriggs_01_gpt_4o_research_agent.ipynb` | GPT-4o powered research agent |
| `agentcon_workflow_automation.py` | AgentCon workflow demo |

### Sub-projects
- **`langgraph-harishneel1/`** — structured course (Introduction → Reflection → ReAct → HITL → Multi-agent → Streaming)
- **`langgraphgroq/`** — LangGraph examples using Groq as the LLM backend
- **`open_deep_research-langcahin-ai/`** — production deep research agent (has its own CLAUDE.md and README)

## Key Concept
LangGraph models agent logic as a directed graph where nodes are Python functions and edges are routing conditions. `StateGraph` holds shared state between nodes; `END` terminates the graph. This is the recommended pattern for multi-step, multi-tool agents over plain LangChain chains.
