# LangChain Examples

LangChain code samples from multiple contributors, covering fundamentals through advanced RAG and agent patterns.

## Setup
```bash
conda env create -f environment.yml
conda activate langchain
```

## Contents

### Cookbooks (recommended starting point)
- `langchain Cookbook Part 1 - Fundamentals_gkamradt.ipynb` — chains, prompts, models, output parsers
- `langchain Cookbook Part 2 - Use Cases_gkamradt.ipynb` — QA, summarization, extraction, evaluation

### 101 Series (colinmcnamara) — Streamlit integrations
Notebooks `langchain_101-1` through `langchain_103` build a streaming document-search chat UI step by step.

### v1 API Scripts
Scripts prefixed `langchain_v1_` use the current LangChain v0.2+ API (LCEL, `RunnablePassthrough`, etc.):
- `langchain_v1_lcel.py` — LangChain Expression Language chains
- `langchain_v1_models.py` — LLM and chat model wrappers
- `langchain_v1_memory.py` — conversation memory
- `langchain_v1_agent.py` / `langchain_v1_createagent.py` — ReAct agent creation

### Advanced Topics
- `langchain_Advanced_RAG_lucifertrj.ipynb` — hybrid search, reranking
- `langchain_Embedchain_RAG_lucifertrj.ipynb` — Embedchain-based RAG
- `langchain_Agents_SQL_Database_Agent_Nichite.ipynb` — SQL agent
- `langchain_dair-ai-prompt_engg-lecture.ipynb` — prompt engineering patterns

## Note on API versions
Files named `langchain_v1_*` use the current LCEL-based API. Older notebooks (colinmcnamara series) use the legacy `LLMChain` pattern — they still run but the API is deprecated.
