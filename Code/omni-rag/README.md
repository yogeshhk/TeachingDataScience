# Omni-RAG: Multimodal Retrieval Augmented Generation

This project implements a research-grade RAG system capable of understanding complex document structures (tables, headers) using a Markdown-first approach.

## 🚀 Features
- **Docling Integration:** Parses PDFs to Markdown, preserving table schemas for LLM understanding.
- **LangGraph Agent:** Orchestrates retrieval and generation in a stateful graph.
- **Groq Acceleration:** Uses `gemma2-9b-it` for near-instant inference.
- **Ragas Evaluation:** Automated metrics for Faithfulness and Context Precision.

## 🛠️ Setup

1. **Clone the repo**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt```
   
3. Set Environment Variables: Create a .env file: ```GROQ_API_KEY=your_groq_api_key_here```

4. Run the UI: ```streamlit run app.py```

## Evaluation

To run the Ragas evaluation suite: ```python evaluate.py```