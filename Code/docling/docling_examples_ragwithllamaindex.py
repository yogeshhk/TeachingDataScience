# https://docling-project.github.io/docling/examples/rag_llamaindex/
import os
from pathlib import Path
from tempfile import mkdtemp
from warnings import filterwarnings
import torch

from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
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

# SOLUTION 1: Use a better model for text generation/summarization
GEN_MODEL = HuggingFaceLLM(
    model_name="google/flan-t5-base",  # Better for Q&A and summarization
    tokenizer_name="google/flan-t5-base",
    context_window=2048,
    max_new_tokens=512,  # Increased token limit
    model_kwargs={
        "torch_dtype": torch.float32,
        "do_sample": True,
        "temperature": 0.3,  # Lower temperature for more focused responses
        "top_p": 0.95,
        "repetition_penalty": 1.1,  # Prevent repetition
    },
    tokenizer_kwargs={
        "padding_side": "left",
        "truncation": True,
        "max_length": 2048,
    },
    device_map="auto" if torch.cuda.is_available() else "cpu",
)

# SOLUTION 2: Alternative - Use GPT-2 which is better for text generation
# GEN_MODEL = HuggingFaceLLM(
#     model_name="gpt2-medium",  # Better text generation capabilities
#     tokenizer_name="gpt2-medium",
#     context_window=1024,
#     max_new_tokens=512,
#     model_kwargs={
#         "torch_dtype": torch.float32,
#         "do_sample": True,
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "pad_token_id": 50256,
#         "eos_token_id": 50256,
#     },
#     tokenizer_kwargs={
#         "padding_side": "left",
#         "pad_token": "<|endoftext|>",
#     },
#     device_map="cpu",
# )

SOURCE = "data/NVIDIAAn.pdf"  # Docling Technical Report
QUERY = "What is NVIDIA's outlook for the first quarter of fiscal 2026?"

embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))
reader = DoclingReader()
node_parser = MarkdownNodeParser()

# EphemeralClient operates purely in-memory, PersistentClient will also save to disk
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# construct vector store
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection,
)

index = VectorStoreIndex.from_documents(
    documents=reader.load_data(SOURCE),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)

# SOLUTION 3: Configure query engine with specific parameters
query_engine = index.as_query_engine(
    llm=GEN_MODEL,
    similarity_top_k=3,  # Retrieve top 3 most relevant chunks
    response_mode="compact",  # Use compact mode for better context utilization
)

result = query_engine.query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
print([(n.text, n.metadata) for n in result.source_nodes])