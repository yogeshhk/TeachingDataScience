"""
Response Generator
Generates natural language answers from retrieved context
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional
# from openai import OpenAI
from groq import Groq  # Instead of: from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from tools.financial_calculator import FinancialCalculator

load_dotenv()


@dataclass
class GeneratedResponse:
    """Response from LLM with metadata"""
    answer: str
    sources: List[str]
    confidence: str
    model_used: str
    tokens_used: Optional[int] = None


class ResponseGenerator:
    """
    LLM powered response generator
    Takes retrieved chunks and generates natural language answers
    """
    
    def __init__(
        self,
        model: str = "gemma2-9b-it",
        temperature: float = None,
        api_key: str = None
    ):
        """
        Initialize GPT-4o generator
        
        Args:
            model: Model name (default from .env or 'gemma2-9b-it')
            temperature: Sampling temperature (default from .env or 0.1)
            api_key: LLM API key (default from .env)
        """
        # Load from environment
        # self.model = model or os.getenv('LLM_MODEL', 'gpt-4o')
        api_key = os.getenv("GROQ_API_KEY")  # Instead of: OPENAI_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")  # Update error message
        
        self.client = Groq(api_key=api_key)  # Instead of: OpenAI(api_key=api_key)
        self.model = model
        print(f"  ✓ Groq Generator initialized (Model: {model})")  # Update message
        
        self.temperature = temperature or float(os.getenv('LLM_TEMPERATURE', '0.1'))
        # api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "LLM API key not found!\n"
                "Set API_KEY in .env file or pass as argument"
            )
        
        # Initialize OpenAI client
        # self.client = OpenAI(api_key=api_key)
        
        # Initialize calculator
        self.calculator = FinancialCalculator()
        
        print(f"✓ Response Generator initialized")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Calculator tools: Available")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_chunks: int = 5
    ) -> GeneratedResponse:
        """
        Generate answer from retrieved chunks
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            max_chunks: Maximum chunks to include in context
            
        Returns:
            GeneratedResponse with answer and metadata
        """
        # Prepare context from chunks
        context = self._format_context(retrieved_chunks[:max_chunks])
        
        # Create system prompt
        system_prompt = self._get_system_prompt()
        
        # Create user prompt
        user_prompt = self._format_user_prompt(query, context)
        
        # Call GPT-4o
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=1000,  # Groq uses max_tokens (not max_completion_tokens)
            top_p=0.9
        )
        
        # Extract response
        answer = response.choices[0].message.content
        
        # Extract sources (page numbers)
        sources = self._extract_sources(retrieved_chunks[:max_chunks])
        
        # Determine confidence
        confidence = self._estimate_confidence(retrieved_chunks[:max_chunks])
        
        return GeneratedResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            model_used=self.model,
            tokens_used=response.usage.total_tokens
        )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for financial Q&A"""
        return """You are a helpful AI assistant specializing in mutual fund analysis.
    
Your task: Answer questions accurately using ONLY the provided factsheet context.

Guidelines:
- Use ONLY information from the context provided
- Be precise with numbers (AUM, returns, ratios, holdings)
- Cite specific fund names when relevant
- If information is not in context, say "Information not available in factsheet"
- For comparisons, present data in clear tables or bullet points
- Keep answers concise but complete
- Use professional financial terminology

Your role:
- Answer questions about mutual fund factsheets accurately
- Use ONLY the provided context - never make up information
- Cite page numbers when available
- For numerical data, provide exact values from the context
- For tables, preserve the structure when showing data
- **IMPORTANT:** If a table shows "NA" for a metric, explain that the fund doesn't have enough history (e.g., fund is too new for 3-year returns)
- Be concise but complete

**Special Cases:**
1. **AUM Changes:** If asked about monthly AUM changes, the factsheet typically shows:
   - "Month end AUM" (current month-end value)
   - "AAUM" (Average AUM for the month)
   - The difference between AAUM and Month-end AUM can indicate inflows/outflows
   - If previous month data is not available, explain what IS available (current AUM and AAUM) and calculate the difference if helpful

2. **Missing Data:** If information is not in the context, say "This specific information is not available in the factsheet" and suggest what related information IS available

**Calculator Tools Available:**
You can perform financial calculations such as:
- CAGR (Compound Annual Growth Rate)
- Absolute returns and percentage changes
- Future value projections
- SIP calculations
- Expense ratio impact
- Fund comparisons

If a user asks for calculations (e.g., "Calculate CAGR for 3 years"), extract the necessary data from context and perform the calculation, then show your work.

Format:
- Start with a direct answer (or explain what data is/isn't available)
- Include relevant details from tables if applicable
- If data shows "NA", explain WHY (fund inception date, insufficient history, etc.)
- For calculations, show the formula and steps
- End with source citations (Page X)
"""
    
    def _format_user_prompt(self, query: str, context: str) -> str:
        """Format user prompt with query and context"""
        return f"""Based on the following context from a mutual fund factsheet, please answer the question.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_type = chunk.get('chunk_type', 'unknown')
            page = chunk.get('page_number', 'N/A')
            fund = chunk.get('fund_name', 'N/A')
            content = chunk.get('content', '')
            
            # Format based on chunk type
            if chunk_type == 'table':
                context_parts.append(
                    f"[Source {i} - Page {page}] TABLE:\n{content}\n"
                )
            elif chunk_type == 'metadata':
                context_parts.append(
                    f"[Source {i} - Page {page}] METADATA:\n{content}\n"
                )
            else:
                context_parts.append(
                    f"[Source {i} - Page {page}]:\n{content}\n"
                )
        
        return "\n---\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract source citations from chunks"""
        sources = []
        for chunk in chunks:
            page = chunk.get('page_number', 'N/A')
            chunk_type = chunk.get('chunk_type', 'text')
            sources.append(f"Page {page} ({chunk_type})")
        
        return list(dict.fromkeys(sources))  # Remove duplicates while preserving order
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]]) -> str:
        """Estimate confidence based on retrieval scores"""
        if not chunks:
            return "Low"
        
        # Get average score
        scores = [chunk.get('score', 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.8:
            return "High"
        elif avg_score > 0.6:
            return "Medium"
        else:
            return "Low"


def main():
    """Test GPT-4o generator"""
    
    print(f"{'='*70}")
    print("GPT-4o GENERATOR TEST")
    print(f"{'='*70}\n")
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in .env")
        print("\nTo test:")
        print("1. Add to .env: OPENAI_API_KEY=sk-...")
        print("2. Re-run this test")
        return
    
    # Initialize generator
    try:
        generator = ResponseGenerator()
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        return
    
    # Mock retrieved chunks (normally from retriever)
    mock_chunks = [
        {
            'chunk_id': 'meta_fund_001',
            'chunk_type': 'metadata',
            'page_number': 2,
            'fund_name': 'Bajaj Finserv Large Cap Fund',
            'content': """Fund: Bajaj Finserv Large Cap Fund
Category: Large Cap Fund
Inception Date: 20-Aug-2024""",
            'score': 0.92
        },
        {
            'chunk_id': 'table_015',
            'chunk_type': 'table',
            'page_number': 15,
            'fund_name': 'Bajaj Finserv Large Cap Fund',
            'content': """Table: Fund Details
| Metric | Value |
|--------|-------|
| AUM | ₹1,610.77 crores |
| Category | Large Cap Fund |
| Benchmark | Nifty 100 TRI |""",
            'score': 0.88
        }
    ]
    
    # Test query
    query = "What is the AUM of Bajaj Finserv Large Cap Fund?"
    
    print(f"Query: {query}\n")
    print("Generating response...")
    
    try:
        response = generator.generate(query, mock_chunks)
        
        print(f"\n{'='*70}")
        print("GENERATED RESPONSE")
        print(f"{'='*70}\n")
        
        print(f"Answer:\n{response.answer}\n")
        print(f"Confidence: {response.confidence}")
        print(f"Sources: {', '.join(response.sources)}")
        print(f"Model: {response.model_used}")
        print(f"Tokens: {response.tokens_used}")
        
    except Exception as e:
        print(f"❌ Error generating: {e}")
        return
    
    print(f"\n{'='*70}")
    print("✅ GPT-4o Generator Working!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
