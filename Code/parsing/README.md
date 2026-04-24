# AI-Powered Resume Analysis Toolkit

This project provides a comprehensive toolkit for parsing resumes and building a Retrieval-Augmented Generation (RAG) system to chat with the resume data. It uses modern NLP and Generative AI techniques.

## Features

- **Multiple Parsing Options**: Includes parsers using the `docling` library and the `GROQ` API with Google's Gemma.
- **Advanced RAG Pipeline**: Implements a RAG system using LlamaIndex, FAISS for vector storage, and Groq's Llama 3 for generation.
- **Semantic Chunking**: Employs semantic splitting for more context-aware document chunking.
- **Interactive Chat UI**: A Streamlit-based web interface for easy file uploading and querying.

## File Descriptions

- `llm_parsing_docling.py`: A class for parsing resumes using the Docling API.
- `llm_parsing_groq.py`: A class that uses the Groq API (with Gemma) for JSON-based resume parsing.
- `llm_llamaindex_rag.py`: Builds a RAG pipeline by processing resumes, chunking them, creating a FAISS vector index, and setting up a query engine with Llama 3.
- `streamlit_main.py`: The main application file that creates a Streamlit UI for uploading resumes and interacting with the RAG system.
- `requirements.txt`: A list of all the Python dependencies required for this project.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Set up API Keys:**
    You need an API key from [Groq](https://console.groq.com/keys). It is recommended to set this as an environment variable:
    ```bash
    export GROQ_API_KEY='your_groq_api_key'
    ```
    If you plan to use the `docling` parser, you will also need a Docling API key.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_main.py
    ```

5.  **Use the application:**
    -   Open your web browser to the URL provided by Streamlit.
    -   Enter your Groq API key in the sidebar.
    -   Upload resume files (txt, pdf, docx).
    -   Click "Build RAG System".
    -   Ask questions about the resumes in the chat interface.

## Disclaimer

This is a demonstration project. Ensure you have the rights to use and process the resume data you upload. API keys should be handled securely and not be hard-coded directly in the source files, especially in a production environment. Use environment variables or a secret management system.