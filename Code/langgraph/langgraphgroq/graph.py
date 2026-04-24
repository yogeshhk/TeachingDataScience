from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="qwen/qwen3-32b", temperature=0)

class LLMState(TypedDict):
    question: str
    answer: str

def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = f"Answer the following question concisely:\n\n{question}"
    response = model.invoke(prompt)
    return {"question": question, "answer": response.content}


graph = StateGraph(LLMState)
graph.add_node("llm_qa", llm_qa)
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

workflow = graph.compile()

initial_state: LLMState = {"question": "AI future in India.", "answer": ""}
result = workflow.invoke(initial_state)
print("Question:", result["question"])
print("Answer:", result["answer"])