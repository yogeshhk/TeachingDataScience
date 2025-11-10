"""
RAG Evaluation Script using Ragas

This script loads a benchmark test suite from a CSV file, runs the RAG system
implemented in langchain_rag.py against the questions, and calculates RAG
evaluation metrics using the Ragas library.

NOTE: Ragas requires an LLM (e.g., OpenAI, Anthropic, or a robust self-hosted LLM)
for evaluation. Please ensure the necessary API key (e.g., OPENAI_API_KEY) is set
in your environment for Ragas to function correctly.
"""
import os
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path
from langchain_groq import ChatGroq

# --- Configuration and Setup ---
# Set the necessary environment variable for Ragas evaluation LLMs (using OpenAI as standard)
# You must set your API key in the environment for this to work. GROQ_API_KEY

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevance,
        context_recall,
        context_precision,
    )
    from langchain_rag import RAGPipeline, DATA_DIR
    from langchain_core.documents import Document
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

except ImportError as e:
    print(f"Error: Required libraries not found. Please install them: pip install pandas datasets ragas")
    print(f"You may also need to install dependencies for the RAG system (transformers, torch, etc.).")
    raise e
except Exception as e:
    # Handles if the RAGPipeline cannot be imported due to missing internal dependencies 
    # (like docling_parsing components or uninitialized models).
    print(f"Warning: Could not fully import RAGPipeline from langchain_rag.py due to internal dependencies: {e}")
    # We will proceed with a mocked function for demonstration if the RAG system fails to initialize
    
    
BENCHMARK_FILE = "testsuite.csv"
DUMMY_DOCUMENT_PATH = str(Path("data") / "SampleReport.pdf")


def get_rag_response_and_context(rag_pipeline: RAGPipeline, question: str) -> Tuple[str, List[str]]:
    """
    Executes the RAG chain and attempts to extract both the final answer and the contexts.

    NOTE: The original RAGPipeline.query only returns the answer. This is a hack/wrapper
    to simulate the required context retrieval for Ragas evaluation by accessing
    the underlying retriever and re-running the retrieval step.
    
    In a real production system, the RAG chain in 'langchain_rag.py' would be structured 
    to return a dictionary containing both 'answer' and 'contexts' in a single invocation.
    """
    try:
        # Step 1: Retrieve documents using the RAG system's retriever
        retrieved_docs: List[Document] = rag_pipeline.rag_system.retriever.get_relevant_documents(question)

        # Ragas expects a list of strings for contexts, which are the content of the retrieved documents
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Step 2: Get the final generated answer (calls the full chain)
        answer = rag_pipeline.query_document(question)
        
        return answer, contexts

    except Exception as e:
        logger.error(f"Error during RAG execution for question '{question}': {e}")
        # Return mock data if RAG execution fails to prevent Ragas from crashing immediately
        return "RAG system execution failed.", ["Mock context for failed execution."]


def run_ragas_evaluation():
    """
    Main function to run the RAG evaluation pipeline.
    """

    try:
        # 1. Load the Benchmark Dataset
        df = pd.read_csv(BENCHMARK_FILE)
        df = df.rename(columns={'Answer': 'ground_truth'})
        logger.info(f"Loaded {len(df)} questions from {BENCHMARK_FILE}.")

        # 2. Initialize and Prepare RAG System
        pipeline = RAGPipeline()
        
        # Ensure the 'data' directory exists for RAG caching
        Path("data").mkdir(exist_ok=True) 

        # The RAG system requires ingestion/loading of documents first
        # We assume a dummy PDF exists or the chunks cache is already populated
        if os.path.exists(DUMMY_DOCUMENT_PATH) or (DATA_DIR / "chunks.json").exists():
             pipeline.process_document(DUMMY_DOCUMENT_PATH)
        else:
            logger.error(f"Cannot run RAG. Please create '{DUMMY_DOCUMENT_PATH}' or run 'langchain_rag.py' to populate the cache.")
            return

        # 3. Generate RAG Outputs (Answer and Contexts)
        answers = []
        contexts = []
        questions = df['Question'].tolist()
        
        logger.info("Running RAG system for all benchmark questions...")
        
        for i, question in enumerate(questions):
            logger.debug(f"Query {i+1}/{len(questions)}: {question}")
            answer, context = get_rag_response_and_context(pipeline, question)
            answers.append(answer)
            contexts.append(context)
            
        df['answer'] = answers
        df['contexts'] = contexts
        
        # 4. Create the Ragas Dataset
        ragas_dataset = Dataset.from_pandas(df)
        
        logger.info("RAG outputs generated and formatted into Ragas Dataset.")

        # 5. Define and Run Ragas Evaluation
        metrics = [
            faithfulness,
            answer_relevance,
            context_recall,
            context_precision,
        ]
        
        logger.info(f"Starting Ragas evaluation with metrics: {[m.name for m in metrics]}...")
        
        # Ragas uses the environment's LLM (e.g., OpenAI) for scoring
        result = evaluate(
            ragas_dataset, 
            metrics=metrics, 
            raise_on_error=False
        )
        
        # 6. Output Accuracy Numbers (Results)
        results_df = result.to_pandas()
        
        print("\n" + "="*50)
        print("                 RAGAS EVALUATION RESULTS")
        print("="*50)
        print(result) # Print the summary of metrics
        print("\nDetailed results per question saved to 'ragas_evaluation_details.csv'")
        print("="*50)
        
        # Optional: Save detailed results
        results_df.to_csv("ragas_evaluation_details.csv", index=False)
        
    except Exception as e:
        logger.error(f"An error occurred during Ragas evaluation: {e}")
        print(f"\nFATAL ERROR: Ragas evaluation failed. Check logs for details: {e}")

if __name__ == "__main__":
    run_ragas_evaluation()