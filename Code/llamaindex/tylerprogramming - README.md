
# LlamaIndex Document Query System

This project demonstrates a simple document querying system using LlamaIndex, a powerful tool for building AI-powered applications with large language models.

## What Will We Build?

- We will build a simple document querying system that allows you to ask questions about a set of documents and get answers.
- We will use a language model to answer questions.
- We will use a vector store index to store and query the documents.
- Create a RAG application (Query Engine)
- Create a simple chatbot (ChatEngine)
- Create the engine code first, then the chatbot loop to talk to the engine
```python
while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")
```
- Observability (Logging, Metrics, Traces) - https://docs.llamaindex.ai/en/stable/module_guides/observability/
- Documents vs. Nodes
- LlamaParse
- LlamaHub
- LlamaCloud (is a waitlist, but promising features)
- NEEDED:
    - OPENAI_API_KEY
    - PHOENIX_API_KEY (https://llamatrace.com/login)


## Overview

The main script (`llamaindex/main.py`) performs the following tasks:

1. Loads documents from a PDF directory
2. Creates or loads a vector store index
3. Executes a query on the indexed documents

## Dependencies

- LlamaIndex
- OpenAI API (for the underlying language model)

## Code Explanation
### Environment Setup

The project uses `pip` to manage dependencies. To install the required packages, run:

```bash
pip install llama-index openai
```

### Loading Documents

The `load_documents` function in `main.py` uses `SimpleDirectoryReader` from LlamaIndex to load all PDF files from the specified directory.

### Creating or Loading a Vector Store Index

The `create_index` function in `main.py` creates a new vector store index if it doesn't exist, otherwise, it loads the existing index.

### Query Execution

The `query_index` function in `main.py` uses the OpenAI API to execute a query on the indexed documents.

## Usage

To run the project, execute the following command:

```bash
python main.py
```

This will load the documents, create or load the index, and execute the query.


