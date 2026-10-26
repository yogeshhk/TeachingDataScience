# LlamaIndex Examples

Scripts and notebooks covering LlamaIndex fundamentals, RAG patterns, agents, and integrations with Groq, Ollama, and HuggingFace.

## Setup

Create the conda environment from `environment.yml` (it installs `llama-index` with the Groq and
HuggingFace integrations), then set the key the script needs:

```bash
conda env create -f environment.yml
conda activate llamaindex
set GROQ_API_KEY=your_key_here
```

## Structure

```
llamaindex/
├── data/                        # Sample documents for RAG demos
├── data_query_modules/          # Saved index stores (JSON)
├── storage/                     # Persisted index storage
├── pdf/                         # PDF exports of some notebooks
├── *.py                         # Standalone scripts
└── *.ipynb                      # Tutorial notebooks
```

## Key Files

| File | Description |
|------|-------------|
| `cookbook_llama3_with_groq.py` | LlamaIndex + Llama 3 via Groq: basic RAG pipeline |
| `llamaindex_cookbook_ollama_mistral_llamaindex.ipynb` | Local RAG using Ollama + Mistral |
| `docs_llamaindex_building_agent.py` | Agent built with LlamaIndex |
| `docx_llamaindex_text2sql_quide.py` | Text-to-SQL guide (the filename spells "quide") |
| `llamaindex_deep_research_agent.py` | Deep-research agent script |
| `Workflows_walkthrough.ipynb` | LlamaIndex Workflows API walkthrough |
| `llamaindex_runllama_1_llms.ipynb` to `llamaindex_runllama_5_retrieval.ipynb` | Five-part series: LLMs, documents and nodes, evaluation baseline, embeddings, retrieval |
| `tylerprogramming - 01 ...` to `07 ...` | Seven short scripts (get running, chat engine, chatbot, RAG pipeline, observability, basic agent, LlamaParse), with their own `tylerprogramming - README.md` |

## Topics Covered

- Document ingestion and chunking
- Vector stores (in-memory and persistent)
- Query engines and chat engines
- Agents
- Text-to-SQL
- Evaluation
