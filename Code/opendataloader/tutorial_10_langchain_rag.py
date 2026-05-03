"""
Tutorial 10 — LangChain RAG Pipeline
Covers: OpenDataLoaderPDFLoader, splitting, FAISS vector store, Q&A retrieval.

Install extras first:
    pip install langchain-opendataloader-pdf langchain langchain-community faiss-cpu

Run:  python tutorial_10_langchain_rag.py
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# ── Load PDFs via LangChain loader ────────────────────────────────────────────
try:
    from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader
except ImportError:
    print("Install: pip install langchain-opendataloader-pdf")
    raise

print("Loading PDFs with OpenDataLoaderPDFLoader ...")
loader = OpenDataLoaderPDFLoader(
    file_path=[
        str(DATA_DIR / "Systems Thinking in 25 Words or Less.pdf"),
        str(DATA_DIR / "Systems Thinking Four Key Questions.pdf"),
    ],
    format="text",
)
docs = loader.load()
print(f"Loaded {len(docs)} document chunk(s)")
for d in docs[:2]:
    src  = d.metadata.get("source", "?")
    page = d.metadata.get("page", "?")
    print(f"  source={Path(src).name}  page={page}  chars={len(d.page_content)}")

# ── Split into smaller chunks ─────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"\nAfter splitting: {len(chunks)} chunks")

# ── Build FAISS vector store with a lightweight embeddings model ──────────────
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print("\nBuilding FAISS index (HuggingFace all-MiniLM-L6-v2) ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"Index built with {vector_store.index.ntotal} vectors")

    # ── Retrieval test ────────────────────────────────────────────────────────
    QUESTIONS = [
        "What is systems thinking?",
        "What are the four key questions in systems thinking?",
        "How does feedback relate to systems thinking?",
    ]

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("\n" + "=" * 60)
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        results = retriever.invoke(q)
        for i, r in enumerate(results, 1):
            src  = Path(r.metadata.get("source", "?")).name
            page = r.metadata.get("page", "?")
            snippet = r.page_content.replace("\n", " ")[:120]
            print(f"  [{i}] {src} p.{page}: {snippet}")

except ImportError as exc:
    print(f"\nSkipping vector-store step: {exc}")
    print("Install: pip install langchain-community faiss-cpu sentence-transformers")

# ── Demonstrate loader with all PDFs in folder ────────────────────────────────
print("\n" + "=" * 60)
print("Loading all PDFs in folder ...")
all_loader = OpenDataLoaderPDFLoader(
    file_path=[str(DATA_DIR)],
    format="text",
)
all_docs = all_loader.load()
print(f"Total chunks from folder: {len(all_docs)}")
total_chars = sum(len(d.page_content) for d in all_docs)
print(f"Total characters        : {total_chars:,}")
