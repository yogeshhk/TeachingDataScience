# LlamaIndex Examples

Scripts and notebooks covering LlamaIndex fundamentals, RAG patterns, agents, and integrations with Groq, Ollama, and HuggingFace.

## Setup

```bash
pip install llama-index llama-index-llms-groq llama-index-embeddings-huggingface
export GROQ_API_KEY=your_key_here
```

## Structure

```
llamaindex/
├── data/                        # Sample documents for RAG demos
├── data_query_modules/          # Reusable query engine helpers
├── *.py                         # Standalone scripts
└── *.ipynb                      # Tutorial notebooks
```

## Key Files

| File | Description |
|------|-------------|
| `cookbook_llama3_with_groq.py` | LlamaIndex + Llama 3 via Groq — basic RAG pipeline |
| `cookbook_ollama_mistral.ipynb` | Local RAG using Ollama + Mistral |
| `building_agent.py` | ReAct agent built with LlamaIndex |
| `docx_llamaindex_text2sql_guide.py` | Text-to-SQL over a Word document |
| `Workflows_walkthrough.ipynb` | LlamaIndex Workflows API walkthrough |

## Topics Covered

- Document ingestion and chunking
- Vector stores (in-memory and persistent)
- Query engines and chat engines
- ReAct and function-calling agents
- Text-to-SQL
- Evaluation with Ragas
