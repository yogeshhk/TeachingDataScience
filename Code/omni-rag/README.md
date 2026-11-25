# Omni-RAG: Multimodal Retrieval Augmented Generation

A research-grade RAG system that understands complex document structures (tables, headers) using a Markdown-first approach.

## Features

- **Docling integration:** parses PDFs to Markdown, preserving table schemas for the LLM.
- **LangGraph agent:** orchestrates retrieval and generation in a stateful graph.
- **Groq acceleration:** uses `gemma2-9b-it` for fast inference.
- **Ragas evaluation:** automated metrics for faithfulness and context precision.

## Setup

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your key:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. Run the UI:

   ```bash
   streamlit run app.py
   ```

## Evaluation

Run the Ragas evaluation suite:

```bash
python evaluate.py
```
