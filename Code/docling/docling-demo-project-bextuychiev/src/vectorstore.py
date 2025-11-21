"""
Vector store management for document storage and retrieval.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """Manages document chunking, embedding, and vector storage."""

    def __init__(self):
        """Initialize the vector store manager."""
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        print(f"✂️ Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create a Chroma vector store from document chunks.

        Args:
            chunks: List of document chunks

        Returns:
            Chroma vector store instance
        """
        print(f"🔢 Creating vector store with {len(chunks)} chunks...")

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="documents"
            )
            print("✅ Vector store created successfully")
            return vectorstore

        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise

    def search_similar(self, vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
        """
        Perform semantic similarity search.

        Args:
            vectorstore: The Chroma vector store
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        try:
            results = vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"❌ Error searching vector store: {str(e)}")
            return []
