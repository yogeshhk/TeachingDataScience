# Author: Rajib Deb
# A simple example showing how langgraph works
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph


def call_bbc_agent(state):
    messages = state['messages']
    print("bbc ", messages)
    # I have hard coded the below. But I can very well call a open ai model to get the response
    response = "Here is the India news from BBC"
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def call_cnn_agent(state):
    messages = state['messages']
    print("cnn ", messages)
    # I have hard coded the below. But I can very well call a bedrock model to get the response
    response = "Here is the India news from CNN"
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def call_ndtv_agent(state):
    messages = state['messages']
    print("ndtv ", messages)
    # I have hard coded the below. But I can very well call a gemini model to get the response
    response = "Here is the India news from NDTV"
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def call_fox_agent(state):
    messages = state['messages']
    print("fox ", messages)
    # I have hard coded the below. But I can very well call a LlaMa model to get the response
    response = "Here is the India news from FOX"
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("bbc_agent", call_bbc_agent)
workflow.add_node("cnn_agent", call_cnn_agent)
workflow.add_node("ndtv_agent", call_ndtv_agent)
workflow.add_node("fox_agent", call_fox_agent)

workflow.set_entry_point("bbc_agent")
# bbc_agent->cnn_agent
workflow.add_edge('bbc_agent', 'cnn_agent')
# bbc_agent->cnn_agent->ndtv_agent
workflow.add_edge('cnn_agent', 'ndtv_agent')
# bbc_agent->cnn_agent->ndtv_agent->fox_agent
workflow.add_edge('ndtv_agent', 'fox_agent')
workflow.set_finish_point("fox_agent")

app = workflow.compile()

inputs = {"messages": [HumanMessage(content="What is India news")]}
response = app.invoke(inputs)
messages = response["messages"]
for message in messages:
    if isinstance(message, HumanMessage):
        print("Question :", message.content)
    else:
        print(message)
