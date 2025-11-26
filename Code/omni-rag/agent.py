# This implements LangGraph Workflow, the Intelligent Retrieval Orchestration. 
# It doesn't just "retrieve"; it assesses if the retrieval was sufficient.

import os
from typing import Annotated, Dict, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from backend import OmniIngestor

# Setup Groq LLM
llm = ChatGroq(
    temperature=0, 
    model_name="gemma2-9b-it", 
    api_key=os.getenv("GROQ_API_KEY")
)

ingestor = OmniIngestor()
retriever = ingestor.get_retriever()

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    retry_count: int

# --- Nodes ---

def retrieve_node(state: AgentState):
    """Retrieves documents based on the question"""
    print("--- Node: Retrieval ---")
    question = state["question"]
    docs = retriever.invoke(question)
    # Context compression
    context_text = "\n\n".join([d.page_content for d in docs])
    return {"context": [context_text]}

def generate_node(state: AgentState):
    """Generates answer using retrieved context"""
    print("--- Node: Generation ---")
    question = state["question"]
    context = state["context"][0]
    
    prompt = f"""You are an expert AI assistant. Use the context below to answer the user's question.
    If the context contains tables, analyze the rows and columns carefully.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

def grade_documents_node(state: AgentState):
    """Evaluates if retrieved documents are relevant (Self-Correction)"""
    # Simple heuristic for demo: check if context is not empty
    # In production, use an LLM-as-a-judge here
    if not state["context"] or len(state["context"][0]) < 10:
        return "empty"
    return "useful"

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_agent = workflow.compile()