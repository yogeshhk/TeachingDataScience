# This implements LangGraph Workflow, the Intelligent Retrieval Orchestration. 
# It doesn't just "retrieve"; it assesses if the retrieval was sufficient.

import os
from typing import Annotated, Dict, TypedDict
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatLlamaCpp
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from backend import OmniIngestor

# Setup Groq LLM
# llm = ChatGroq(
#     temperature=0,
#     model_name="gemma2-9b-it",
#     api_key=os.getenv("GROQ_API_KEY")
# )

# Local Qwen3-1.7B via llama.cpp, in-process, no server (see
# Code/langchain/langchain_v1_models.py for the validated reference pattern, and
# CLAUDE.md's "Local LLM (Qwen3)" section for the pilot findings). CPU-only,
# ~17-24 tok/s on this machine. This graph never calls bind_tools() (retrieval is a
# direct Python call to `retriever.invoke()`, not an LLM-bound tool), so it is
# unaffected by the bind_tools() limitation documented for the two genuinely
# tool-calling scripts.
llm = ChatLlamaCpp(
    model_path=r"D:\Yogesh\models\lmstudio-community\Qwen3-1.7B-GGUF\Qwen3-1.7B-Q4_K_M.gguf",
    n_ctx=4096,
    temperature=0,
    max_tokens=1000,
    n_gpu_layers=0,
    verbose=False,
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

    Answer: /no_think"""
    # /no_think: Qwen3 soft-switch, skips its default "thinking" mode which otherwise
    # consumes the max_tokens budget before producing an answer. See CLAUDE.md.

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