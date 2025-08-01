# https://docling-project.github.io/docling/examples/rag_llamaindex/
import os
from pathlib import Path
from tempfile import mkdtemp
from warnings import filterwarnings
import torch

from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM  # Changed this import
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.core import VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

filterwarnings(action="ignore", category=UserWarning, module="pydantic")
filterwarnings(action="ignore", category=FutureWarning, module="easyocr")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")

# Use local model instead of Inference API
GEN_MODEL = HuggingFaceLLM(
    model_name="microsoft/DialoGPT-small",  # Small model that works locally
    tokenizer_name="microsoft/DialoGPT-small",
    context_window=1024,
    max_new_tokens=256,
    model_kwargs={
        "torch_dtype": torch.float32,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": 50256,  # Add pad token for DialoGPT
    },
    tokenizer_kwargs={
        "padding_side": "left",
    },
    device_map="cpu",  # Use CPU to avoid GPU memory issues
)

SOURCE = "data/NVIDIAAn.pdf"  # Docling Technical Report
QUERY = "What is NVIDIA’s outlook for the first quarter of fiscal 2026?"

# embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))
# reader = DoclingReader()
# node_parser = MarkdownNodeParser()

# # EphemeralClient operates purely in-memory, PersistentClient will also save to disk
# chroma_client = chromadb.EphemeralClient()
# chroma_collection = chroma_client.create_collection("quickstart")

# # construct vector store
# vector_store = ChromaVectorStore(
#     chroma_collection=chroma_collection,
# )

# index = VectorStoreIndex.from_documents(
#     documents=reader.load_data(SOURCE),
#     transformations=[node_parser],
#     storage_context=StorageContext.from_defaults(vector_store=vector_store),
#     embed_model=EMBED_MODEL,
# )

result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
print([(n.text, n.metadata) for n in result.source_nodes])