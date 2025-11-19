"""
Milvus/Zilliz Cloud Vector Storage
Stores embeddings and metadata for semantic search
"""
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MilvusStorage:
    """
    Milvus/Zilliz Cloud vector database for semantic search
    
    Collection Schema:
    - chunk_id (primary key)
    - embedding (vector, dim=384)
    - chunk_type (metadata, table, text, image)
    - page_number
    - fund_name
    - content (full text)
    - word_count
    """
    
    def __init__(
        self,
        collection_name: str = "bajaj_factsheet_chunks",
        embedding_dim: int = 384,
        use_zilliz_cloud: bool = True,  # Default to cloud
        uri: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize Milvus connection
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            use_zilliz_cloud: Whether to use Zilliz Cloud (True) or local Milvus (False)
            uri: Zilliz Cloud URI (if None, loads from .env)
            token: Zilliz Cloud token (if None, loads from .env)
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Load credentials from .env if not provided
        if use_zilliz_cloud:
            uri = uri or os.getenv('MILVUS_URI') or os.getenv('ZILLIZ_CLOUD_URI')
            token = token or os.getenv('MILVUS_TOKEN') or os.getenv('ZILLIZ_CLOUD_TOKEN')
            
            if not uri or not token:
                raise ValueError(
                    "Zilliz Cloud credentials not found!\n"
                    "Please set MILVUS_URI and MILVUS_TOKEN in .env file\n"
                    "Or pass them as arguments: MilvusStorage(uri='...', token='...')"
                )
        
        # Connect to Milvus/Zilliz Cloud
        if use_zilliz_cloud:
            print(f"Connecting to Zilliz Cloud...")
            print(f"  URI: {uri[:30]}..." if uri else "  URI: Not set")
            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )
        else:
            print(f"Connecting to local Milvus...")
            connections.connect(
                alias="default",
                host='localhost',
                port='19530'
            )
        
        print(f"✓ Connected to Milvus")
        
        # Create or load collection
        self.collection = self._create_collection()
        print(f"✓ Collection ready: {collection_name}")
    
    def _create_collection(self) -> Collection:
        """Create or get collection with schema"""
        
        # Check if collection exists
        if utility.has_collection(self.collection_name):
            print(f"Loading existing collection: {self.collection_name}")
            collection = Collection(self.collection_name)
            collection.load()
            return collection
        
        print(f"Creating new collection: {self.collection_name}")
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="fund_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="word_count", dtype=DataType.INT64),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Bajaj Finserv Factsheet Chunks with Embeddings"
        )
        
        # Create collection
        collection = Collection(
            name=self.collection_name,
            schema=schema
        )
        
        # Create index for vector search
        index_params = {
            "metric_type": "COSINE",  # Use cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        print(f"✓ Created index on embedding field")
        
        # Load collection into memory
        collection.load()
        
        return collection
    
    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Insert chunks with embeddings into Milvus
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        print(f"Inserting {len(chunks)} chunks...")
        
        # Prepare data for insertion
        chunk_ids = []
        embeddings = []
        chunk_types = []
        page_numbers = []
        fund_names = []
        contents = []
        word_counts = []
        
        for chunk in chunks:
            chunk_ids.append(chunk['chunk_id'])
            embeddings.append(chunk['embedding'])
            chunk_types.append(chunk['chunk_type'])
            page_numbers.append(chunk['page_number'])
            fund_names.append(chunk.get('fund_name', '') or '')
            
            # Truncate content if too long
            content = chunk['content']
            if len(content) > 65000:
                content = content[:65000]
            contents.append(content)
            
            word_counts.append(chunk.get('metadata', {}).get('word_count', 0))
        
        # Insert data
        entities = [
            chunk_ids,
            embeddings,
            chunk_types,
            page_numbers,
            fund_names,
            contents,
            word_counts
        ]
        
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        
        print(f"✓ Inserted {insert_result.insert_count} chunks")
        return insert_result
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using query embedding
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_expr: Optional filter expression (e.g., "chunk_type == 'table'")
            
        Returns:
            List of search results with scores
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Ensure query embedding is 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        results = self.collection.search(
            data=query_embedding.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["chunk_id", "chunk_type", "page_number", "fund_name", "content", "word_count"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'chunk_id': hit.entity.get('chunk_id'),
                    'score': hit.score,
                    'chunk_type': hit.entity.get('chunk_type'),
                    'page_number': hit.entity.get('page_number'),
                    'fund_name': hit.entity.get('fund_name'),
                    'content': hit.entity.get('content'),
                    'word_count': hit.entity.get('word_count'),
                    'distance': hit.distance
                })
        
        return formatted_results
    
    def search_by_fund(
        self,
        query_embedding: np.ndarray,
        fund_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within specific fund"""
        filter_expr = f'fund_name == "{fund_name}"'
        return self.search(query_embedding, top_k, filter_expr)
    
    def search_by_type(
        self,
        query_embedding: np.ndarray,
        chunk_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search within specific chunk type"""
        filter_expr = f'chunk_type == "{chunk_type}"'
        return self.search(query_embedding, top_k, filter_expr)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        stats = {
            'name': self.collection_name,
            'num_entities': self.collection.num_entities,
            'schema': str(self.collection.schema),
        }
        return stats
    
    def delete_collection(self):
        """Delete the collection"""
        utility.drop_collection(self.collection_name)
        print(f"✓ Deleted collection: {self.collection_name}")
    
    def close(self):
        """Close connection"""
        connections.disconnect("default")
        print(f"✓ Disconnected from Milvus")


def main():
    """Test Milvus storage"""
    
    print(f"{'='*70}")
    print("MILVUS VECTOR STORAGE SETUP")
    print(f"{'='*70}\n")
    
    # For local testing, we'll use Milvus Lite (embedded Milvus)
    # Install with: pip install milvus
    
    try:
        from milvus import default_server
        
        # Start local Milvus server
        print("Starting local Milvus server...")
        default_server.start()
        print("✓ Milvus server started\n")
        
        # Initialize storage
        storage = MilvusStorage(
            collection_name="bajaj_factsheet_test",
            embedding_dim=384,
            use_zilliz_cloud=False
        )
        
        # Load embeddings
        print("\nLoading embeddings...")
        with open('data/processed/chunks_with_embeddings.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data['chunks']
        print(f"✓ Loaded {len(chunks)} chunks with embeddings")
        
        # Insert chunks
        storage.insert_chunks(chunks[:50])  # Insert first 50 for testing
        
        # Get statistics
        print(f"\n{'='*70}")
        print("COLLECTION STATISTICS")
        print(f"{'='*70}\n")
        
        stats = storage.get_stats()
        print(f"Collection: {stats['name']}")
        print(f"Total entities: {stats['num_entities']}")
        
        # Test search
        print(f"\n{'='*70}")
        print("TESTING SEARCH")
        print(f"{'='*70}\n")
        
        # Use first chunk's embedding as query
        test_embedding = np.array(chunks[0]['embedding'])
        
        results = storage.search(test_embedding, top_k=3)
        
        print(f"Search results for test query:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Chunk: {result['chunk_id']}")
            print(f"   Type: {result['chunk_type']}")
            print(f"   Page: {result['page_number']}")
            print(f"   Content: {result['content'][:100]}...")
        
        # Cleanup
        print(f"\n{'='*70}")
        
        # Don't delete for now - keep for testing
        # storage.delete_collection()
        
        storage.close()
        default_server.stop()
        
        print("✅ Milvus storage test complete!")
        print(f"{'='*70}")
        
    except ImportError:
        print("⚠ Milvus Lite not installed. Install with: pip install milvus")
        print("\nAlternatively, use Zilliz Cloud:")
        print("  1. Sign up at: https://cloud.zilliz.com/")
        print("  2. Create a cluster")
        print("  3. Get your URI and API token")
        print("  4. Set in .env file:")
        print("     ZILLIZ_CLOUD_URI=your_uri")
        print("     ZILLIZ_CLOUD_TOKEN=your_token")
        print("\nFor now, we'll use FAISS as a fallback for local development.")


if __name__ == "__main__":
    main()
