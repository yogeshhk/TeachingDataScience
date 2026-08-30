# This module handles the heavy lifting: parsing PDFs into a "Hybrid Representation" 
# where tables are treated as structured Markdown and standard text is chunked normally.

import os
from typing import List
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

class OmniIngestor:
    def __init__(self):
        self.converter = DocumentConverter()
        # Using a lightweight, high-performance embedding model suitable for RAG
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = Chroma(
            collection_name="omni_rag_v1",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Multimodal Parsing: Extracts Text, Tables (as Markdown), and Image Metadata
        """
        print(f"--- 1. Parsing Document: {file_path} ---")
        result = self.converter.convert(file_path)
        doc = result.document
        
        # Export full content to Markdown (preserves table structure better than plain text)
        full_markdown = doc.export_to_markdown()
        
        # Strategy: Split by Headers to preserve semantic sections
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits = markdown_splitter.split_text(full_markdown)
        
        # Enrich metadata for hybrid retrieval
        for split in splits:
            split.metadata["source"] = file_path
            # Heuristic: If snippet contains pipe characters, it's likely a table
            if "|" in split.page_content and "-|-" in split.page_content:
                split.metadata["type"] = "table"
            else:
                split.metadata["type"] = "text"
                
        print(f"--- 2. Ingesting {len(splits)} Chunks into Vector Store ---")
        self.vector_store.add_documents(splits)
        return splits

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_type="mmr", # Maximum Marginal Relevance for diversity
            search_kwargs={"k": 5}
        )