from typing import TypedDict
from langgraph.graph import END, StateGraph

class SimpleState(TypedDict):
    count: int


def increment(state: SimpleState) -> SimpleState: 
    return {
        "count": state["count"] + 1
    }

def should_continue(state):
    if(state["count"] < 5): 
        return "continue"
    else: 
        return "stop"
    
graph = StateGraph(SimpleState)

graph.add_node("increment", increment)

graph.set_entry_point("increment")

graph.add_conditional_edges(
    "increment", 
    should_continue, 
    {
        "continue": "increment", 
        "stop": END
    }
)

app = graph.compile()

state = {
    "count": 0
}

result = app.invoke(state)
print(result)
