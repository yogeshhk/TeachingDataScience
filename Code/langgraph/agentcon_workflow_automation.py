from typing import List, Dict
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

class EmailState(BaseModel):
    email_content: str = ""
    is_spam: bool = False
    category: str = ""
    draft_response: str = ""
    messages: List[Dict[str, str]] = []
  
# 
llm = ChatGroq(model="llama-3.1-8b-instant") 

def read_email(state: EmailState) -> EmailState:
    return state

def classify_email(state: EmailState) -> EmailState:
    prompt = f"Classify this email as spam or legitimate:\n{state.email_content}\nRespond with only 'spam' or 'legitimate'."
    response = llm.invoke(prompt)
    is_spam = "spam" in response.content.lower()
    return EmailState(
        email_content=state.email_content,
        is_spam=is_spam,
        category=state.category,
        draft_response=state.draft_response,
        messages=state.messages + [{"role": "classifier", "content": response.content}]
    )


def handle_spam(state: EmailState) -> EmailState:
    new_messages = state.messages + [{"role": "system", "content": "Email marked as spam"}]
    return EmailState(
        email_content=state.email_content,
        is_spam=state.is_spam,
        category=state.category,
        draft_response=state.draft_response,
        messages=new_messages
    )

def draft_response(state: EmailState) -> EmailState:
    prompt = f"Draft a professional response to:\n{state.email_content}"
    response = llm.invoke(prompt)
    return EmailState(
        email_content=state.email_content,
        is_spam=state.is_spam,
        category=state.category,
        draft_response=response.content,
        messages=state.messages + [{"role": "drafter", "content": response.content}]
    )
        
def route_email(state: EmailState) -> str:
    """Conditional routing based on classification"""
    if state.is_spam:
        return "spam_handler"
    else:
        return "response_drafter"
    
# Create graph
workflow = StateGraph(EmailState)

# Add nodes
workflow.add_node("read_email", read_email)
workflow.add_node("classify_email", classify_email)
workflow.add_node("spam_handler", handle_spam)
workflow.add_node("response_drafter", draft_response)

# Add edges
workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_email")

def route_email(state: EmailState) -> str:
    if state.is_spam:
        return "spam_handler"  # Exact node name
    else:
        return "response_drafter"  # Exact node name

workflow.add_conditional_edges("classify_email", route_email)

workflow.add_edge("spam_handler", END)
workflow.add_edge("response_drafter", END)

# Compile
app = workflow.compile()


# Visualize and save the graph
png_graph = app.get_graph().draw_mermaid_png()
with open("langgraph_email_workflow_graph.png", "wb") as f:
    f.write(png_graph)
print(f"Graph saved as 'langgraph_email_workflow_graph.png' in {os.getcwd()}") 

# Test with legitimate email
state = EmailState(
    email_content="Hello, I'd like to schedule a meeting to discuss the project timeline."
)
result = app.invoke(state)
print(f"Is Spam: {result['is_spam']}")
print(f"Draft Response: {result['draft_response']}")

# Test with spam email
spam_state = EmailState(
    email_content="URGENT! You've won $1M! Click here now!!!"
)
spam_result = app.invoke(spam_state)
print(f"Is Spam: {spam_result['is_spam']}")