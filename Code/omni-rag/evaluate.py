# Ragas Metrics script runs a quantitative evaluation of the RAG pipeline, 
# focusing on how well it handles the retrieved context.

import os
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from backend import OmniIngestor
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize components for Ragas
# Ragas needs an LLM for "LLM-as-a-Judge" metrics
eval_llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def run_evaluation():
    ingestor = OmniIngestor()
    retriever = ingestor.get_retriever()
    
    # 1. Create a Synthetic Test Set (Golden Dataset)
    # In a real scenario, these are hand-verified Q&A pairs
    questions = [
        "What is the total revenue in Q3 according to the table?",
        "List the safety protocols defined in section 2."
    ]
    ground_truths = [
        ["The total revenue in Q3 was $1.5M."],
        ["Wear helmets, safety goggles, and steel-toed boots."]
    ]
    
    # 2. Run the Pipeline
    answers = []
    contexts = []
    
    llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"))
    
    for query in questions:
        # Retrieval
        retrieved_docs = retriever.invoke(query)
        context_str = [doc.page_content for doc in retrieved_docs]
        contexts.append(context_str)
        
        # Generation
        prompt = f"Context: {context_str}\n\nQuestion: {query}"
        ans = llm.invoke(prompt).content
        answers.append(ans)
        
    # 3. Prepare Dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Execute Evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\n--- Evaluation Results ---")
    print(results)
    return results

if __name__ == "__main__":
    run_evaluation()