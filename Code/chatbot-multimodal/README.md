# Multi-Modal RAG with Docling

A RAG system for documents that mix text, tables, images and code blocks. Docling parses the document
into typed chunks, each chunk gets a short description, and only the descriptions are embedded for
retrieval. Table questions are answered by converting the query to SQL; the other chunk types are
returned as text or description.

## Files

| File | Purpose |
|---|---|
| `docling_parsing.py` | `DoclingParser`: parse a document into Pydantic `TextChunk`, `TableChunk`, `ImageChunk`, `CodeChunk` objects and generate a description for each; caches chunks in `data/chunks.json` |
| `llamaindex_rag.py` | `MultiModalRAG` and `RAGPipeline` (`process_document`, `query_document`): LlamaIndex retrieval over the descriptions, with a `TableQueryAgent` for table questions |
| `langchain_rag.py` | The same idea on LangChain with a FAISS index (cached in `data/faiss_index`) |
| `ragas_rag.py` | Evaluates the LangChain RAG against `data/testsuite.csv` with Ragas; needs an evaluation LLM key such as `OPENAI_API_KEY` |
| `test_models.py` | Mocked pytest suite |
| `data/` | `SampleReport.pdf`, `testsuite.csv`, and the cached `chunks.json` and `faiss_index` |
| `environment.yml`, `requirements.txt` | Dependencies (conda environment `chatbot-multimodal`) |
| `RAG_Document_Parsing_Report.md`, `Multi-modal Partitioning for RAG_Medium.pdf` | Background reading |

## Run

```bash
conda env create -f environment.yml
conda activate chatbot-multimodal
set GROQ_API_KEY=your_key_here
python llamaindex_rag.py
python langchain_rag.py
python -m pytest test_models.py -v
```

```python
from llamaindex_rag import RAGPipeline

pipeline = RAGPipeline()
pipeline.process_document("data/SampleReport.pdf")
print(pipeline.query_document("What are the main topics discussed?"))
```

## Configuration

- Text descriptions: `DoclingParser` defaults to `gemma2-9b-it` through Groq and reads `GROQ_API_KEY`;
  without the key it falls back to simpler descriptions.
- Image descriptions: `microsoft/git-base`; `device` is `"auto"`, `"cpu"` or `"cuda"`.
- Retrieval in `llamaindex_rag.py`: embeddings `sentence-transformers/all-MiniLM-L6-v2`, generation
  `microsoft/DialoGPT-small`; both are constructor arguments of `MultiModalRAG`.
- Delete `data/chunks.json` and `data/faiss_index` to force re-parsing of a changed document.
