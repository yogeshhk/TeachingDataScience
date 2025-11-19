"""
FAISS Vector Storage (Local Fallback)
Fast similarity search without requiring a server
"""
import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional


class FAISSStorage:
    """
    FAISS-based vector storage for local development
    Lightweight alternative to Milvus/Zilliz Cloud
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: str = "data/processed/faiss_index.bin",
        metadata_path: str = "data/processed/faiss_metadata.pkl"
    ):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load chunk metadata
        """
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        # Create FAISS index
        # Using IndexFlatIP for cosine similarity (Inner Product after normalization)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks_metadata = []
        self.chunk_id_map = {}
        
        # Load existing index if available
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
        
        print(f"✓ FAISS index ready (dim={embedding_dim})")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks with embeddings to index
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        print(f"Adding {len(chunks)} chunks to FAISS index...")
        
        # Extract embeddings
        embeddings = np.array([chunk['embedding'] for chunk in chunks], dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        start_idx = len(self.chunks_metadata)
        self.index.add(embeddings)
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            metadata = {
                'idx': start_idx + i,
                'chunk_id': chunk['chunk_id'],
                'chunk_type': chunk['chunk_type'],
                'page_number': chunk['page_number'],
                'fund_name': chunk.get('fund_name', ''),
                'content': chunk['content'],
                'word_count': chunk.get('metadata', {}).get('word_count', 0)
            }
            self.chunks_metadata.append(metadata)
            self.chunk_id_map[chunk['chunk_id']] = start_idx + i
        
        print(f"✓ Added {len(chunks)} chunks (total: {len(self.chunks_metadata)})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_fund: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_fund: Optional fund name filter
            filter_type: Optional chunk type filter
            
        Returns:
            List of results with scores
        """
        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search - get more results if filtering
        search_k = top_k * 10 if (filter_fund or filter_type) else top_k
        distances, indices = self.index.search(query, search_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            metadata = self.chunks_metadata[idx].copy()
            metadata['score'] = float(dist)  # Cosine similarity score
            
            # Apply filters
            if filter_fund and metadata['fund_name'] != filter_fund:
                continue
            if filter_type and metadata['chunk_type'] != filter_type:
                continue
            
            results.append(metadata)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_fund(
        self,
        query_embedding: np.ndarray,
        fund_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within specific fund"""
        return self.search(query_embedding, top_k, filter_fund=fund_name)
    
    def search_by_type(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within specific chunk type"""
        return self.search(query_embedding, top_k, filter_type=chunk_type)
    
    def save(self):
        """Save index and metadata to disk"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'chunks_metadata': self.chunks_metadata,
                'chunk_id_map': self.chunk_id_map
            }, f)
        
        print(f"✓ Saved FAISS index to: {self.index_path}")
        print(f"✓ Saved metadata to: {self.metadata_path}")
    
    def load(self):
        """Load index and metadata from disk"""
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks_metadata = data['chunks_metadata']
            self.chunk_id_map = data['chunk_id_map']
        
        print(f"✓ Loaded FAISS index from: {self.index_path}")
        print(f"  Total chunks: {len(self.chunks_metadata)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        chunk_types = {}
        funds = set()
        
        for meta in self.chunks_metadata:
            chunk_types[meta['chunk_type']] = chunk_types.get(meta['chunk_type'], 0) + 1
            if meta['fund_name']:
                funds.add(meta['fund_name'])
        
        return {
            'total_chunks': len(self.chunks_metadata),
            'embedding_dim': self.embedding_dim,
            'chunk_types': chunk_types,
            'unique_funds': len(funds)
        }


def main():
    """Test FAISS storage"""
    
    print(f"{'='*70}")
    print("FAISS VECTOR STORAGE SETUP")
    print(f"{'='*70}\n")
    
    # Initialize storage
    storage = FAISSStorage(embedding_dim=384)
    
    # Load embeddings
    print("Loading embeddings...")
    with open('data/processed/chunks_with_embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    print(f"✓ Loaded {len(chunks)} chunks with embeddings")
    
    # Add chunks to index
    storage.add_chunks(chunks)
    
    # Save index
    storage.save()
    
    # Get statistics
    print(f"\n{'='*70}")
    print("INDEX STATISTICS")
    print(f"{'='*70}\n")
    
    stats = storage.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Unique funds: {stats['unique_funds']}")
    print(f"\nChunk types:")
    for ctype, count in stats['chunk_types'].items():
        print(f"  {ctype}: {count}")
    
    # Test search
    print(f"\n{'='*70}")
    print("TESTING SEARCH")
    print(f"{'='*70}\n")
    
    # Test queries
    from src.embeddings.embedder import EmbeddingGenerator
    
    embedder = EmbeddingGenerator()
    
    test_queries = [
        ("What is the NAV of Bajaj Finserv Large Cap Fund?", None, None),
        ("Show me the top holdings", None, "table"),
        ("Performance of flexi cap fund", "Bajaj Finserv Flexi Cap Fund", None),
    ]
    
    for query, fund_filter, type_filter in test_queries:
        print(f"Query: '{query}'")
        if fund_filter:
            print(f"  Filter fund: {fund_filter}")
        if type_filter:
            print(f"  Filter type: {type_filter}")
        
        query_embedding = embedder.embed_query(query)
        
        if fund_filter:
            results = storage.search_by_fund(query_embedding, fund_filter, top_k=3)
        elif type_filter:
            results = storage.search_by_type(query_embedding, type_filter, top_k=3)
        else:
            results = storage.search(query_embedding, top_k=3)
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.4f}")
            print(f"     Type: {result['chunk_type']} | Page: {result['page_number']} | Fund: {result.get('fund_name', 'N/A')}")
            content = result['content'][:120].replace('\n', ' ')
            print(f"     Content: {content}...")
        print()
    
    print(f"{'='*70}")
    print("✅ FAISS Storage Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    main()
