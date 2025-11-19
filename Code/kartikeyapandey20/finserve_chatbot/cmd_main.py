"""
Bajaj Finserv Mutual Fund Chatbot - Command Line Test Interface
Minimal CLI for testing RAG workflow without UI dependencies
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from rag.langgraph_workflow import RAGWorkflow


class RAGTester:
    """Minimal command-line interface for RAG testing"""
    
    def __init__(self, vector_storage_type='faiss'):
        print("🚀 Initializing RAG workflow...")
        self.rag = RAGWorkflow(vector_storage_type=vector_storage_type)
        print("✅ RAG workflow initialized successfully!\n")
    
    def query(self, question):
        """Execute a single query and display results"""
        print(f"\n{'='*80}")
        print(f"❓ QUERY: {question}")
        print(f"{'='*80}")
        
        try:
            result = self.rag.query(question)
            
            # Display answer
            print(f"\n💬 ANSWER:")
            print(f"{result['answer']}\n")
            
            # Display metadata
            print(f"📊 METADATA:")
            print(f"  📄 Chunks retrieved: {result['num_chunks']}")
            print(f"  🎯 Query type: {result['query_type']}")
            print(f"  ✓ Confidence: {result['confidence']}")
            
            # Display sources
            if result.get('sources'):
                print(f"\n📚 SOURCES:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source}")
            
            print(f"\n{'='*80}\n")
            return result
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}\n")
            print(f"{'='*80}\n")
            return None
    
    def interactive_mode(self):
        """Run in interactive mode - continuous Q&A"""
        print("\n" + "="*80)
        print("💼 BAJAJ FINSERV FUND CHATBOT - INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  • Type your question and press Enter")
        print("  • Type 'exit' or 'quit' to stop")
        print("  • Type 'examples' to see sample questions")
        print("="*80 + "\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if question.lower() == 'examples':
                    self.show_examples()
                    continue
                
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except EOFError:
                print("\n\n👋 Goodbye!\n")
                break
    
    def show_examples(self):
        """Display example questions"""
        examples = [
            "What is the AUM of Large Cap Fund?",
            "Which fund has highest 1-year return?",
            "Show top 5 holdings of Flexi Cap Fund",
            "Compare expense ratios of all funds",
            "What is the risk profile of Small Cap Fund?",
            "Investment philosophy of Large Cap Fund?"
        ]
        
        print("\n" + "="*80)
        print("💡 EXAMPLE QUESTIONS:")
        print("="*80)
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print("="*80 + "\n")
    
    def batch_test(self, questions):
        """Test multiple questions in batch"""
        print("\n" + "="*80)
        print(f"🧪 BATCH TESTING - {len(questions)} QUERIES")
        print("="*80 + "\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            result = self.query(question)
            results.append(result)
        
        return results


def main():
    """Main entry point with different modes"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['-h', '--help', 'help']:
            print("""
Bajaj Finserv Fund Chatbot - CLI Test Interface

Usage:
  python test_rag.py                    # Interactive mode
  python test_rag.py interactive        # Interactive mode (explicit)
  python test_rag.py batch              # Run batch tests
  python test_rag.py query "question"   # Single query
  python test_rag.py help               # Show this help

Examples:
  python test_rag.py query "What is the AUM of Large Cap Fund?"
  python test_rag.py batch
  python test_rag.py
            """)
            return
        
        # Initialize tester
        tester = RAGTester()
        
        if command == 'query' and len(sys.argv) > 2:
            # Single query mode
            question = " ".join(sys.argv[2:])
            tester.query(question)
        
        elif command == 'batch':
            # Batch test mode
            test_questions = [
                "What is the AUM of Large Cap Fund?",
                "Which fund has highest 1-year return?",
                "Show top 5 holdings of Flexi Cap Fund",
                "Compare expense ratios of all funds",
                "What is the risk profile of Small Cap Fund?"
            ]
            tester.batch_test(test_questions)
        
        elif command == 'interactive':
            # Interactive mode
            tester.interactive_mode()
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'python test_rag.py help' for usage information")
    
    else:
        # Default: interactive mode
        tester = RAGTester()
        tester.interactive_mode()


if __name__ == "__main__":
    main()
