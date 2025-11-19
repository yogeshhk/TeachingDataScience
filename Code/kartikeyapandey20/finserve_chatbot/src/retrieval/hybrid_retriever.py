"""
Hybrid Retriever - Combines Vector Search + Structured Queries
Intelligent retrieval system for financial factsheet Q&A
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from storage.sqlite_storage import SQLiteStorage
from storage.storage_factory import VectorStorageFactory
from embeddings.embedder import EmbeddingGenerator


@dataclass
class RetrievalResult:
    """Single retrieval result with metadata"""
    chunk_id: str
    content: str
    score: float
    chunk_type: str
    page_number: int
    fund_name: Optional[str]
    source: str  # 'vector', 'sql', or 'hybrid'
    metadata: Dict[str, Any] = None


class HybridRetriever:
    """
    Hybrid retrieval combining:
    1. FAISS semantic search (for general questions)
    2. SQLite structured queries (for numerical/factual data)
    3. Re-ranking and fusion
    """
    
    def __init__(
        self,
        vector_storage_type: str = "faiss",
        vector_storage: Any = None,
        sql_storage: SQLiteStorage = None,
        embedder: EmbeddingGenerator = None
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_storage_type: 'faiss' or 'milvus' (auto-loads from factory)
            vector_storage: Pre-initialized vector storage (optional)
            sql_storage: SQLite structured storage (optional, auto-loads)
            embedder: Embedding generator (optional, auto-loads)
        """
        # Use storage factory if not provided
        if vector_storage is None:
            if vector_storage_type:
                # Use specific type
                self.vector_storage = VectorStorageFactory.create(vector_storage_type)
            else:
                # Auto-detect from .env
                self.vector_storage = VectorStorageFactory.get_default()
        else:
            self.vector_storage = vector_storage
        
        # Load SQLite storage
        if sql_storage is None:
            self.sql_storage = SQLiteStorage()
        else:
            self.sql_storage = sql_storage
        
        # Load embedder
        if embedder is None:
            self.embedder = EmbeddingGenerator()
        else:
            self.embedder = embedder
        
        print("✓ Hybrid Retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fund_filter: Optional[str] = None,
        use_hybrid: bool = True
    ) -> List[RetrievalResult]:
        """
        Main retrieval method
        
        Args:
            query: User query
            top_k: Number of results to return
            fund_filter: Optional fund name filter
            use_hybrid: Whether to use hybrid retrieval (True) or just vector search (False)
            
        Returns:
            List of RetrievalResult objects
        """
        # Detect query type
        query_type = self._classify_query(query)
        
        if use_hybrid and query_type in ['numerical', 'factual']:
            # Use hybrid retrieval for structured queries
            return self._hybrid_retrieve(query, top_k, fund_filter, query_type)
        else:
            # Use pure vector search for semantic queries
            return self._vector_retrieve(query, top_k, fund_filter)
    
    def _classify_query(self, query: str) -> str:
        """
        Classify query type
        
        Types:
        - numerical: NAV, AUM, returns, performance metrics
        - factual: holdings, allocations, fund details
        - semantic: general questions, explanations
        - calculation: CAGR, comparisons, changes
        """
        query_lower = query.lower()
        
        # Numerical keywords
        numerical_keywords = ['nav', 'aum', 'return', 'returns', 'performance', 
                             'growth', 'yield', 'ratio', 'beta', 'alpha']
        
        # Factual keywords
        factual_keywords = ['holding', 'holdings', 'allocation', 'sector', 
                           'top', 'companies', 'stocks', 'portfolio']
        
        # Calculation keywords
        calculation_keywords = ['cagr', 'compare', 'difference', 'change', 
                               'versus', 'vs', 'better', 'higher', 'lower']
        
        # Check for keywords
        if any(kw in query_lower for kw in calculation_keywords):
            return 'calculation'
        elif any(kw in query_lower for kw in numerical_keywords):
            return 'numerical'
        elif any(kw in query_lower for kw in factual_keywords):
            return 'factual'
        else:
            return 'semantic'
    
    def _vector_retrieve(
        self,
        query: str,
        top_k: int,
        fund_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """Pure vector search"""
        
        # Encode query
        query_embedding = self.embedder.embed_query(query)
        
        # Search
        if fund_filter:
            results = self.vector_storage.search_by_fund(
                query_embedding, fund_filter, top_k
            )
        else:
            results = self.vector_storage.search(query_embedding, top_k)
        
        # Convert to RetrievalResult
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                chunk_id=result['chunk_id'],
                content=result['content'],
                score=result['score'],
                chunk_type=result['chunk_type'],
                page_number=result['page_number'],
                fund_name=result.get('fund_name'),
                source='vector',
                metadata={'word_count': result.get('word_count', 0)}
            ))
        
        return retrieval_results
    
    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        fund_filter: Optional[str],
        query_type: str
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining vector + SQL
        
        Strategy:
        1. Extract fund name from query if not provided
        2. Get vector search results
        3. Get SQL results based on query type
        4. Merge and re-rank using RRF (Reciprocal Rank Fusion)
        """
        
        # Extract fund name if not provided
        if not fund_filter:
            fund_filter = self._extract_fund_name(query)
        
        # Vector search results
        query_embedding = self.embedder.embed_query(query)
        
        if fund_filter:
            vector_results = self.vector_storage.search_by_fund(
                query_embedding, fund_filter, top_k * 2
            )
        else:
            vector_results = self.vector_storage.search(query_embedding, top_k * 2)
        
        # SQL results based on query type
        sql_results = []
        
        if query_type == 'numerical' and fund_filter:
            # Get fund metadata
            fund_data = self.sql_storage.get_fund(fund_filter)
            if fund_data:
                sql_results.append({
                    'chunk_id': f'sql_fund_{fund_filter}',
                    'content': self._format_fund_metadata(fund_data),
                    'score': 1.0,
                    'chunk_type': 'metadata',
                    'page_number': fund_data.get('page_number', 0),
                    'fund_name': fund_filter,
                    'source': 'sql'
                })
        
        elif query_type == 'factual' and fund_filter:
            # Get tables for fund
            tables = self.sql_storage.get_tables_by_fund(fund_filter)
            for table in tables[:3]:  # Top 3 tables
                # Filter out None values from headers
                headers = [h for h in table['headers'][:5] if h is not None]
                headers_str = ', '.join(str(h) for h in headers) if headers else 'N/A'
                
                sql_results.append({
                    'chunk_id': table['table_id'],
                    'content': f"Table from page {table['page_number']}\n" +
                              f"Rows: {table['row_count']}, Cols: {table['col_count']}\n" +
                              f"Headers: {headers_str}",
                    'score': 0.9,
                    'chunk_type': 'table',
                    'page_number': table['page_number'],
                    'fund_name': fund_filter,
                    'source': 'sql'
                })
        
        # Merge and re-rank using Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            vector_results, sql_results, top_k
        )
        
        return merged_results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        sql_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) for combining results
        
        RRF Score = Σ 1 / (k + rank_i)
        where k = 60 is a constant, rank_i is the rank in result set i
        """
        
        # Calculate RRF scores
        chunk_scores = {}
        chunk_data = {}
        
        # Vector results
        for rank, result in enumerate(vector_results, 1):
            chunk_id = result['chunk_id']
            rrf_score = 1 / (k + rank)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0
                chunk_data[chunk_id] = result
            
            chunk_scores[chunk_id] += rrf_score
        
        # SQL results
        for rank, result in enumerate(sql_results, 1):
            chunk_id = result['chunk_id']
            rrf_score = 1 / (k + rank)
            
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0
                chunk_data[chunk_id] = result
            
            chunk_scores[chunk_id] += rrf_score
        
        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            data = chunk_data[chunk_id]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=rrf_score,
                chunk_type=data['chunk_type'],
                page_number=data['page_number'],
                fund_name=data.get('fund_name'),
                source=data.get('source', 'hybrid'),
                metadata={}
            ))
        
        return results
    
    def _extract_fund_name(self, query: str) -> Optional[str]:
        """Extract fund name from query"""
        
        # Get all funds from database
        funds = self.sql_storage.get_all_funds()
        
        # Check if any fund name appears in query
        query_lower = query.lower()
        for fund in funds:
            fund_name = fund['fund_name']
            if fund_name.lower() in query_lower:
                return fund_name
            
            # Check for partial matches
            # e.g., "large cap" should match "Bajaj Finserv Large Cap Fund"
            fund_keywords = fund_name.lower().replace('bajaj finserv', '').replace('fund', '').strip().split()
            if any(keyword in query_lower for keyword in fund_keywords if len(keyword) > 3):
                return fund_name
        
        return None
    
    def _format_fund_metadata(self, fund_data: Dict[str, Any]) -> str:
        """Format fund metadata as text"""
        lines = [f"Fund: {fund_data['fund_name']}"]
        
        if fund_data.get('category'):
            lines.append(f"Category: {fund_data['category']}")
        if fund_data.get('aum_crores'):
            lines.append(f"AUM: ₹{fund_data['aum_crores']:.2f} crores")
        if fund_data.get('inception_date'):
            lines.append(f"Inception Date: {fund_data['inception_date']}")
        if fund_data.get('benchmark'):
            lines.append(f"Benchmark: {fund_data['benchmark']}")
        if fund_data.get('nav'):
            lines.append(f"NAV: ₹{fund_data['nav']:.2f}")
        
        return '\n'.join(lines)
    
    def retrieve_by_fund(
        self,
        fund_name: str,
        query: str = "",
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """Get all information about a specific fund"""
        
        if query:
            # Semantic search within fund
            return self.retrieve(query, top_k, fund_filter=fund_name)
        else:
            # Get all chunks for fund
            chunks = self.sql_storage.get_chunks_by_fund(fund_name)
            
            # If fund has chunks, get their full content from vector storage
            results = []
            for chunk_meta in chunks[:top_k]:
                # Find in vector storage
                chunk_idx = self.vector_storage.chunk_id_map.get(chunk_meta['chunk_id'])
                if chunk_idx is not None:
                    chunk_data = self.vector_storage.chunks_metadata[chunk_idx]
                    results.append(RetrievalResult(
                        chunk_id=chunk_data['chunk_id'],
                        content=chunk_data['content'],
                        score=1.0,
                        chunk_type=chunk_data['chunk_type'],
                        page_number=chunk_data['page_number'],
                        fund_name=chunk_data['fund_name'],
                        source='sql',
                        metadata={}
                    ))
            
            return results


def main():
    """Test hybrid retriever"""
    
    print(f"{'='*70}")
    print("HYBRID RETRIEVER TEST")
    print(f"{'='*70}\n")
    
    # Initialize components
    print("Loading components...")
    
    # Create retriever (auto-loads components)
    retriever = HybridRetriever()
    
    # Test queries
    test_cases = [
        {
            'query': "What is the NAV of Bajaj Finserv Large Cap Fund?",
            'expected_type': 'numerical',
            'description': 'Numerical query with fund name'
        },
        {
            'query': "Show me the top holdings",
            'expected_type': 'factual',
            'description': 'Factual query without fund name'
        },
        {
            'query': "What is the fund manager's investment strategy?",
            'expected_type': 'semantic',
            'description': 'Semantic query'
        },
        {
            'query': "Performance of flexi cap fund in the last year",
            'expected_type': 'numerical',
            'description': 'Performance query with partial fund name'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test['description']}")
        print(f"{'='*50}")
        print(f"Query: {test['query']}")
        
        # Classify query
        query_type = retriever._classify_query(test['query'])
        print(f"Detected type: {query_type} (expected: {test['expected_type']})")
        
        # Extract fund name
        fund_name = retriever._extract_fund_name(test['query'])
        print(f"Extracted fund: {fund_name or 'None'}")
        
        # Retrieve
        results = retriever.retrieve(test['query'], top_k=3)
        
        print(f"\nTop {len(results)} results:")
        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. Score: {result.score:.4f} | Source: {result.source}")
            print(f"   Chunk: {result.chunk_id}")
            print(f"   Type: {result.chunk_type} | Page: {result.page_number}")
            print(f"   Fund: {result.fund_name or 'N/A'}")
            content = result.content[:150].replace('\n', ' ')
            print(f"   Content: {content}...")
    
    print(f"\n{'='*70}")
    print("✅ Hybrid Retriever Working!")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    main()
