# LangGraph Examples

LangGraph code samples covering the core concepts of stateful, graph-based agent orchestration.

## Setup
```bash
conda env create -f environment.yml
conda activate langgraph
```
Set the API keys a script needs (`GROQ_API_KEY`, `GEMINI_API_KEY`) in the environment; there is no
`.env.example`.

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
| `keerti_youtube_introduction.py` | Introductory graph script |

### One place to choose the model, and a ReAct agent from scratch
| File | Concept |
|------|---------|
| `llm_config.py` | The single place to choose the chat model for the seminar examples (`MODEL_NAME`, or the `LLM_MODEL` environment variable). Default: a small local model in LM Studio, no key and no per-token charge. `python llm_config.py` lists the model ids your server offers |
| `react_demo/react_agent.py` | ReAct loop (llm node, tool node, router); the model comes from `llm_config.py`, with `--provider` and `--model` switches |
| `react_demo/text_react_agent.py` | ReAct with a plain-text protocol (`Action:` / `Action Input:` / `Observation:` / `Final Answer:`), no `bind_tools`, for small models that ignore tool schemas |
| `react_demo/workflow_compare.py` | A fixed workflow, not an agent: fetch two cities, compare in code, phrase with the model |
| `workflow_demo/email_triage.py` | Email triage workflow: four nodes, a state with a log reducer, one conditional edge. `python workflow_demo/email_triage.py --email "..."` |
| `test_llm_config.py`, `react_demo/test_*.py`, `workflow_demo/test_email_triage.py` | Mocked tests, no network or keys (`pytest` in each directory) |

Providers: `lmstudio` (default), `ollama`, `groq`, `gemini`. Default setup: load a small model such as
`gemma-2-2b-it` (Q4_K_M, about 1.7 GB) in LM Studio, start its local server, run `python llm_config.py`
and set `MODEL_NAME` to the id it prints. The codebasics notebooks and `langgraphgroq/` keep their own models.

Lesson from a small-model comparison: with a small model, let code decide and let the model write the
sentence (the fixed workflow beat the tool-calling agents).

### Consolidated from Code/agents/ (learning notebooks and CrewAI-in-LangGraph hybrids)
| File | Concept |
|------|---------|
| `LangGraphLearning_menloparklab.ipynb` | General LangGraph learning notes |
| `YT_01_Learning_LangGraph_Agent_Executor_SamWitteveen.ipynb` | Agent executor pattern |
| `YT_02_Learning_LangGraph_Chat_Executor_SamWitteveen.ipynb` | Chat executor pattern |
| `YT_03_Learning_LangGraph_Agent_Supervisor__SamWitteveen.ipynb` | Multi-agent supervisor pattern |
| `langgraph_multiagent.py` | Multi-agent graph example |
| `langgraph_rajibdeb_01_how_to_langgraph_example_01.py` | Basic LangGraph walkthrough |
| `langgraph_rajibdeb_01_how_to_use_crew_ai_for_image_understanding.py` | CrewAI as a tool inside LangGraph (image understanding) |
| `langgraph_rajibdeb_01_how_to_use_crewai_to_solve_math.py` | CrewAI as a tool inside LangGraph (math solving) |
| `langgraph_rajibdeb_03_how_to_persist_shared_state_lg.py` | Shared state persistence |

The two `..._crew*` files need `crewai` (in `environment.yml`); LangGraph is the driving framework, hence
they live here rather than in `Code/crewai/`.

### Sub-project
- **`langgraphgroq/`**: LangGraph example using Groq as the LLM backend (`graph.py`, `langgraph.json`)

## Key Concept
LangGraph models agent logic as a directed graph where nodes are Python functions and edges are routing conditions. `StateGraph` holds shared state between nodes; `END` terminates the graph.
