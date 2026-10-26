# Docling: Book QA and Parsing Examples

A Streamlit app for asking questions about a PDF book (Docling for parsing, LangChain and FAISS for
retrieval, a local LLM server for answers), plus a set of small Docling example scripts and notebooks.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit UI: upload a PDF, chat, view source chunks and the Docling markdown |
| `docling_book_loader.py` | `DoclingBookLoader` and `BookQASystem` classes (parsing, chunking, FAISS index, QA chain) |
| `setup_script.py` | Installs the dependencies into the active environment |
| `AI-Engineer-Skool_booktutor.py` | A separate book-tutor script |
| `docling_examples_*.py` | Small examples: simple and custom conversion, tables and table export, scanned PDFs, LangChain, LlamaIndex RAG |
| `docling.ipynb`, `mayurji_docling-*.ipynb`, `sudarshan_koirala_docling.ipynb` | Tutorial notebooks |
| `docling-demo-project-bextuychiev/` | A third-party demo project with its own `README.md` and `requirements.txt` |
| `data/` | Sample PDFs (including `scanned.pdf` and `table.pdf`) |
| `scratch/` | Exported table CSV and HTML output from the examples |

## Run

Create the conda environment from `environment.yml` and start the app:

```bash
conda env create -f environment.yml
conda activate docling
streamlit run streamlit_app.py
```

The app expects a local OpenAI-compatible LLM server at `http://localhost:1234/v1` (LM Studio, or any
server on that port; see `Code/LOCAL_LLM_PLAN.md`). Start it before asking questions.

## Configuration

Set in `docling_book_loader.py`:

- Docling pipeline: OCR on, table structure on, automatic accelerator with 8 threads
- Chunking: 1000 characters with 200 overlap (`RecursiveCharacterTextSplitter`)
- Retrieval: FAISS, MMR search, 5 chunks per query; the vector store is saved and reused on later loads
- LLM: `ChatOpenAI` with model `local-model`, base URL `http://localhost:1234/v1`, temperature 0

To use another provider, change the `ChatOpenAI` initialisation in `BookQASystem`.

## Tips

- If the Bottleneck "ARRAY_API not found" error appears, rerun `python setup_script.py` in a fresh
  conda environment.
- A connection error means the local LLM server is not running.
- The first load of a book is slow; later loads reuse the saved index.
