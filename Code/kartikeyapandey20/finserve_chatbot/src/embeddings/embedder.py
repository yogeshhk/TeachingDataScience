"""
Embedding Pipeline using sentence-transformers
Generates vector embeddings for all chunks
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using sentence-transformers
    
    Model: all-MiniLM-L6-v2
    - Dimension: 384
    - Fast and efficient
    - Good for semantic search
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name
                - all-MiniLM-L6-v2: 384 dim, fast (default)
                - all-mpnet-base-v2: 768 dim, better quality
                - multi-qa-mpnet-base-dot-v1: 768 dim, optimized for Q&A
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded - Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with added 'embedding' field (numpy array)
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings in batches
        print("Encoding texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Embedding shape: {embeddings.shape}")
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding.tolist()  # Convert to list for JSON
            chunk_copy['embedding_model'] = self.model_name
            chunk_copy['embedding_dim'] = self.embedding_dim
            enhanced_chunks.append(chunk_copy)
        
        return enhanced_chunks
    
    def save_embeddings(self, chunks_with_embeddings: List[Dict[str, Any]], output_path: str):
        """Save chunks with embeddings to JSON"""
        
        output_data = {
            'total_chunks': len(chunks_with_embeddings),
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'generated_at': datetime.now().isoformat(),
            'chunks': chunks_with_embeddings
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved embeddings to: {output_path}")
        
        # Also save embeddings as numpy array for faster loading
        embeddings_array = np.array([c['embedding'] for c in chunks_with_embeddings])
        numpy_path = output_path.replace('.json', '_vectors.npy')
        np.save(numpy_path, embeddings_array)
        print(f"✓ Saved embedding vectors to: {numpy_path}")
    
    def load_embeddings(self, embeddings_path: str) -> tuple:
        """
        Load embeddings from JSON file
        
        Returns:
            (chunks_with_embeddings, embeddings_array)
        """
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data['chunks']
        
        # Try to load numpy array if available
        numpy_path = embeddings_path.replace('.json', '_vectors.npy')
        if Path(numpy_path).exists():
            embeddings_array = np.load(numpy_path)
        else:
            embeddings_array = np.array([c['embedding'] for c in chunks])
        
        return chunks, embeddings_array
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector (numpy array)
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding


def main():
    """Test embedding generation on chunks"""
    
    # Load chunks
    chunks_path = "data/processed/chunks.json"
    print(f"Loading chunks from: {chunks_path}")
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data['chunks']
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Display chunk type distribution
    chunk_types = {}
    for chunk in chunks:
        ctype = chunk['chunk_type']
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
    
    print("\nChunk distribution:")
    for ctype, count in chunk_types.items():
        print(f"  {ctype}: {count}")
    
    # Initialize embedder
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    # Generate embeddings
    chunks_with_embeddings = embedder.generate_embeddings(chunks)
    
    # Save embeddings
    output_path = "data/processed/chunks_with_embeddings.json"
    embedder.save_embeddings(chunks_with_embeddings, output_path)
    
    # Test query embedding
    print(f"\n{'='*70}")
    print("TESTING QUERY EMBEDDING")
    print(f"{'='*70}\n")
    
    test_queries = [
        "What is the NAV of Bajaj Finserv Large Cap Fund?",
        "Show me the top holdings",
        "What is the AUM?",
        "Performance of flexi cap fund"
    ]
    
    print("Sample query embeddings:")
    for query in test_queries:
        query_embedding = embedder.embed_query(query)
        print(f"  Query: '{query}'")
        print(f"    Embedding shape: {query_embedding.shape}")
        print(f"    First 5 values: {query_embedding[:5]}")
        print()
    
    # Simple similarity test
    print(f"{'='*70}")
    print("SIMILARITY TEST")
    print(f"{'='*70}\n")
    
    query = "What is the AUM of Large Cap Fund?"
    query_embedding = embedder.embed_query(query)
    
    print(f"Query: '{query}'")
    print(f"\nTop 5 most similar chunks:\n")
    
    # Calculate cosine similarity with all chunks
    embeddings_array = np.array([c['embedding'] for c in chunks_with_embeddings])
    similarities = np.dot(embeddings_array, query_embedding)
    
    # Get top 5
    top_indices = np.argsort(similarities)[::-1][:5]
    
    for rank, idx in enumerate(top_indices, 1):
        chunk = chunks_with_embeddings[idx]
        score = similarities[idx]
        
        print(f"{rank}. Score: {score:.4f} | Type: {chunk['chunk_type']} | Page: {chunk['page_number']}")
        print(f"   Fund: {chunk.get('fund_name', 'N/A')}")
        content_preview = chunk['content'][:150].replace('\n', ' ')
        print(f"   Content: {content_preview}...\n")
    
    print(f"{'='*70}")
    print("✅ Embedding Pipeline Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
