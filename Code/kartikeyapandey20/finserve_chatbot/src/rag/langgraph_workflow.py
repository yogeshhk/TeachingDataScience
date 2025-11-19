"""
LangGraph RAG Workflow
End-to-end orchestration: Query -> Retrieve -> Generate -> Respond
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import components
from retrieval.hybrid_retriever import HybridRetriever
from src.generation.response_generator import ResponseGenerator, GeneratedResponse
from storage.storage_factory import VectorStorageFactory

load_dotenv()


def check_data_ready() -> bool:
    """Check if required data files exist"""
    required_files = [
        "data/processed/chunks.json",
        "data/processed/chunks_with_embeddings.json",
        "data/processed/chunks_with_embeddings_vectors.npy",
        "data/processed/faiss_index.bin",
        "data/processed/factsheet.db"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("⚠ Missing data files:")
        for file in missing:
            print(f"  - {file}")
        print("\n💡 Run ingestion pipeline first:")
        print("   python src/ingestion/pipeline.py")
        return False
    
    return True


# Define State
class RAGState(TypedDict):
    """State passed through the workflow"""
    query: str
    query_type: str  # numerical/factual/semantic/calculation
    fund_name: str  # Extracted fund name or "all"
    retrieved_chunks: List[Dict[str, Any]]
    generated_response: GeneratedResponse
    final_answer: str
    error: str


class RAGWorkflow:
    """
    LangGraph RAG Workflow
    
    Flow:
    1. Analyze Query -> Classify intent + extract fund
    2. Retrieve Context -> Hybrid FAISS/Milvus + SQLite
    3. Generate Answer -> GPT-4o
    4. Format Response -> Return to user
    """
    
    def __init__(
        self,
        vector_storage_type: str = 'faiss',
        top_k: int = 10,
        model: str = "gemma2-9b-it"  
    ):
        """
        Initialize RAG workflow
        
        Args:
            vector_storage_type: 'faiss' or 'milvus' (default: 'faiss' for speed)
            top_k: Number of chunks to retrieve (increased to 10 for better coverage)
        """
        print(f"{'='*70}")
        print("INITIALIZING RAG WORKFLOW")
        print(f"{'='*70}\n")
        
        # Check if data is ready
        print("0. Checking data files...")
        if not check_data_ready():
            raise RuntimeError(
                "Required data files not found. "
                "Please run: python src/ingestion/pipeline.py"
            )
        print("   ✓ All data files present\n")
        
        # Initialize retriever
        print("1. Setting up Hybrid Retriever...")
        self.retriever = HybridRetriever(vector_storage_type=vector_storage_type)
        
        # Initialize generator
        print("\n2. Setting up Response Generator...")
        self.generator = ResponseGenerator()
        
        self.top_k = top_k
        
        # Build LangGraph
        print("\n3. Building LangGraph workflow...")
        self.workflow = self._build_graph()
        
        print(f"\n{'='*70}")
        print("✓ RAG Workflow Ready!")
        print(f"{'='*70}\n")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        
        # Create graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze", self._analyze_query)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("format", self._format_response)
        
        # Define edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "format")
        workflow.add_edge("format", END)
        
        # Compile
        return workflow.compile()
    
    def _analyze_query(self, state: RAGState) -> RAGState:
        """Step 1: Analyze query"""
        print(f"\n[Step 1] Analyzing query...")
        
        query = state['query']
        
        # Use retriever's built-in query classifier
        query_type = self.retriever._classify_query(query)
        fund_name = self.retriever._extract_fund_name(query)
        
        print(f"  Query Type: {query_type}")
        print(f"  Fund: {fund_name}")
        
        state['query_type'] = query_type
        state['fund_name'] = fund_name
        
        return state
    
    def _retrieve_context(self, state: RAGState) -> RAGState:
        """Step 2: Retrieve relevant chunks"""
        print(f"\n[Step 2] Retrieving context...")
        
        query = state['query']
        
        # Retrieve using hybrid retriever
        results = self.retriever.retrieve(query, top_k=self.top_k)
        
        # Convert RetrievalResult objects to dicts
        chunks = []
        for result in results:
            chunks.append({
                'chunk_id': result.chunk_id,
                'content': result.content,
                'page_number': result.page_number,
                'chunk_type': result.chunk_type,
                'fund_name': result.fund_name,
                'score': result.score,
                'source': result.source
            })
        
        print(f"  Retrieved {len(chunks)} chunks")
        
        # Show preview
        if chunks:
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"    {i}. Page {chunk.get('page_number', '?')} ({chunk.get('chunk_type', '?')}) - Score: {chunk.get('score', 0):.3f}")
        
        state['retrieved_chunks'] = chunks
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Step 3: Generate answer with LLM"""
        print(f"\n[Step 3] Generating answer...")
        
        query = state['query']
        chunks = state['retrieved_chunks']
        
        if not chunks:
            print("  ⚠ No chunks retrieved, generating empty response")
            state['generated_response'] = GeneratedResponse(
                answer="I couldn't find relevant information in the factsheet to answer your question.",
                sources=[],
                confidence="Low",
                model_used="none"
            )
            return state
        
        # Generate with LLM
        response = self.generator.generate(query, chunks, max_chunks=self.top_k)
        
        print(f"  Confidence: {response.confidence}")
        print(f"  Tokens: {response.tokens_used}")
        
        state['generated_response'] = response
        
        return state
    
    def _format_response(self, state: RAGState) -> RAGState:
        """Step 4: Format final response"""
        print(f"\n[Step 4] Formatting response...")
        
        response = state['generated_response']
        
        # Format final answer
        final_answer = f"""{response.answer}

---
**Sources:** {', '.join(response.sources) if response.sources else 'None'}
**Confidence:** {response.confidence}
"""
        
        state['final_answer'] = final_answer
        
        print("  ✓ Response ready")
        
        return state
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Run end-to-end RAG pipeline
        
        Args:
            question: User's question
            
        Returns:
            Dict with answer, sources, confidence
        """
        print(f"\n{'='*70}")
        print(f"QUERY: {question}")
        print(f"{'='*70}")
        
        # Initialize state
        initial_state = {
            'query': question,
            'query_type': '',
            'fund_name': '',
            'retrieved_chunks': [],
            'generated_response': None,
            'final_answer': '',
            'error': ''
        }
        
        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            return {
                'answer': final_state['final_answer'],
                'query_type': final_state['query_type'],
                'fund': final_state['fund_name'],
                'sources': final_state['generated_response'].sources if final_state['generated_response'] else [],
                'confidence': final_state['generated_response'].confidence if final_state['generated_response'] else 'Low',
                'num_chunks': len(final_state['retrieved_chunks'])
            }
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'answer': f"Error processing query: {e}",
                'query_type': 'error',
                'fund': '',
                'sources': [],
                'confidence': 'Low',
                'num_chunks': 0
            }


def main():
    """Test RAG workflow"""
    
    print(f"\n{'='*70}")
    print("LANGGRAPH RAG WORKFLOW - END-TO-END TEST")
    print(f"{'='*70}\n")
    
    # Initialize workflow
    try:
        rag = RAGWorkflow()
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is the AUM of Bajaj Finserv Large Cap Fund?",
        "Which fund has the highest 1-year return?",
        "What is the investment philosophy of Bajaj Finserv Flexi Cap Fund?"
    ]
    
    print(f"\n{'='*70}")
    print(f"RUNNING {len(test_queries)} TEST QUERIES")
    print(f"{'='*70}\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'#'*70}")
        
        result = rag.query(query)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}\n")
        
        print(result['answer'])
        print(f"\nQuery Type: {result['query_type']}")
        print(f"Fund: {result['fund']}")
        print(f"Chunks Retrieved: {result['num_chunks']}")
        
        if i < len(test_queries):
            input("\nPress Enter for next query...")
    
    print(f"\n{'='*70}")
    print("✅ ALL TESTS COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
