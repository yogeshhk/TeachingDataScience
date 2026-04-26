# GraphRAG — Graph-based Retrieval Augmented Generation

Scripts demonstrating knowledge graph construction and graph-based QA using LangChain's `NetworkxEntityGraph` and `GraphQAChain`, with Groq as the LLM backend (via the OpenAI-compatible endpoint).

## Files

| File | Description |
|------|-------------|
| `graphrag_langchain_spicejet.py` | Builds a flight-network graph from CSV and answers route queries |
| `graphrag_langchain_test.py` | Minimal example: builds a graph from plain text sentences and runs QA |

## Setup

```bash
pip install langchain langchain-openai langchain-community networkx pandas
export GROQ_API_KEY=your_key_here
```

## Running

```bash
# Requires data/flight_network_spicejet.csv
python graphrag_langchain_spicejet.py

# Self-contained test with hardcoded text
python graphrag_langchain_test.py
```

## Architecture

```
Text / CSV  →  GraphIndexCreator / manual edges  →  NetworkxEntityGraph
                                                          ↓
                                                    GraphQAChain (LLM)
                                                          ↓
                                                    Natural language answer
```

## Notes

- Both scripts use `ChatOpenAI` pointed at `https://api.groq.com/openai/v1` — a standard pattern for using Groq via the OpenAI SDK.
- Reference: [LangChain GraphRAG article](https://medium.com/data-science-in-your-pocket/graph-analytics-relationship-link-prediction-in-graphs-using-neo4j-79a81716e73a)
