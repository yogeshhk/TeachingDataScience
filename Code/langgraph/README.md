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

### ReAct agent from scratch (Groq or Gemini)
| File | Concept |
|------|---------|
| `gemini/react_agent.py` | ReAct loop (llm node, tool node, router) with a `--provider groq\|gemini` switch |
| `gemini/test_react_agent.py` | Mocked tests, no network or API keys needed (`pytest gemini/`) |

Keys come from `GROQ_API_KEY` / `GEMINI_API_KEY`; model names are constants at the top of `react_agent.py`.
Backs the "ReAct Agent from Scratch with Gemini" section of the LangGraph seminar deck.

### Consolidated from Code/agents/ (learning notebooks + CrewAI-in-LangGraph hybrids)
| File | Concept |
|------|---------|
| `LangGraphLearning_menloparklab.ipynb` | General LangGraph learning notes |
| `YT_01_Learning_LangGraph_Agent_Executor_SamWitteveen.ipynb` | Agent executor pattern |
| `YT_02_Learning_LangGraph_Chat_Executor_SamWitteveen.ipynb` | Chat executor pattern |
| `YT_03_Learning_LangGraph_Agent_Supervisor__SamWitteveen.ipynb` | Multi-agent supervisor pattern |
| `langgraph_multiagent.py` | Multi-agent graph example |
| `langgraph_rajibdeb_01_how_to_langgraph_example_01.py` | Basic LangGraph walkthrough |
| `langgraph_rajibdeb_01_how_to_use_crew_ai_for_image_understanding.py` | CrewAI-as-a-tool inside LangGraph (image understanding) |
| `langgraph_rajibdeb_01_how_to_use_crewai_to_solve_math.py` | CrewAI-as-a-tool inside LangGraph (math solving) |
| `langgraph_rajibdeb_03_how_to_persist_shared_state_lg.py` | Shared state persistence |

The two `..._crew*` files need `crewai` (in `environment.yml`) since they demo CrewAI
running as a tool inside a LangGraph graph — LangGraph is the driving framework, hence
living here rather than in `Code/crewai/`.

### Sub-projects
- **`langgraph-harishneel1/`** — structured course (Introduction → Reflection → ReAct → HITL → Multi-agent → Streaming)
- **`langgraphgroq/`** — LangGraph examples using Groq as the LLM backend
- **`open_deep_research-langcahin-ai/`** — production deep research agent (has its own CLAUDE.md and README)

## Key Concept
LangGraph models agent logic as a directed graph where nodes are Python functions and edges are routing conditions. `StateGraph` holds shared state between nodes; `END` terminates the graph. This is the recommended pattern for multi-step, multi-tool agents over plain LangChain chains.
