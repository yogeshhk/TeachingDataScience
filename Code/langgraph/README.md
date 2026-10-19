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

### One place to choose the model, and a ReAct agent from scratch
| File | Concept |
|------|---------|
| `llm_config.py` | The single place to choose the chat model for the seminar examples (`MODEL_NAME`, or the `LLM_MODEL` environment variable). Default: a small local model in LM Studio, no key and no per-token charge. Run `python llm_config.py` to see the model ids your server offers |
| `test_llm_config.py` | Offline tests for the config (`pytest test_llm_config.py`) |
| `react_demo/react_agent.py` | ReAct loop (llm node, tool node, router); the model comes from `llm_config.py`, with `--provider` and `--model` switches |
| `workflow_demo/email_triage.py` | Email triage workflow for the seminar: four nodes (two plain functions, two LLM calls), a state with a log reducer, one conditional edge. `python workflow_demo/email_triage.py --email "..."`. `read_email` stores the split words in `words_list` |
| `workflow_demo/test_email_triage.py` | Mocked tests with a scripted fake model, no network or keys (`pytest workflow_demo/`) |
| `react_demo/test_react_agent.py` | Mocked tests, no network or API keys needed (`pytest react_demo/`) |
| `react_demo/text_react_agent.py` | ReAct with a plain-text protocol (`Action:` / `Action Input:` / `Observation:` / `Final Answer:`), no `bind_tools`, for small models that ignore tool schemas. An Action beats a Final Answer in the same reply; tool errors come back as Observation text; step limit |
| `react_demo/workflow_compare.py` | A fixed workflow, not an agent: fetch city A, fetch city B, compare in code, phrase with the model. The sentence is checked for the computed numbers, with a template fallback |
| `react_demo/test_text_react_agent.py`, `react_demo/test_workflow_compare.py` | Mocked tests for the two files above |

Measured with `gemma-2-2b-it` (Q4_K_M, LM Studio, temperature 0, real Open-Meteo data, 5 runs each):
`bind_tools` called the tool in 0 of 5 tries; the text ReAct agent was right on a single-city question in
5 of 5 but wrong on a two-city comparison in 0 of 5; the workflow gave a sentence with the right numbers in
5 of 5 (the verdict comes from code). Lesson: with a small model, let code decide and let the model write
the sentence. Run: `python react_demo/text_react_agent.py --question "..."` and `python react_demo/workflow_compare.py`.

Default setup: download `gemma-2-2b-it` (Q4_K_M, about 1.7 GB) in LM Studio, start its local server, run
`python llm_config.py` and set `MODEL_NAME` to the id it prints. Providers: `lmstudio` (default), `ollama`,
`groq`, `gemini`; the last two need `GROQ_API_KEY` / `GEMINI_API_KEY` in the environment. Using the config:
`keerti_youtube_introduction.py`, `agentcon_workflow_automation.py` and `react_demo/react_agent.py`. The
codebasics notebooks and the `langgraphgroq/` project keep their own models.

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
