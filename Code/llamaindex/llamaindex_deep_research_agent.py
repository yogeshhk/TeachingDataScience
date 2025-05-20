# https://www.youtube.com/watch?v=8a_RMSKJC6A
# https://colab.research.google.com/drive/1xYq4wr4dkmvOuq0Ljwt1W9fIxfR50ekq

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from tavily import AsyncTavilyClient
import os
import asyncio # Import the asyncio library
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.core.workflow.errors import WorkflowRuntimeError # Import the specific error

# import nest_asyncio

# nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model
tavily_api_key = os.getenv("TAVILY_API_KEY")

async def search_web(query: str) -> str:
    """ Useful for using web to answer questions"""
    client = AsyncTavilyClient(api_key=tavily_api_key) 
    return str(await client.search(query))          


async def set_name(ctx: Context, name: str) -> str:
    """
    Sets the user's name in the state.
    Args:
        ctx: The context object for accessing state.
        name (str): The name to be set for the user.
    """
    print(f"Attempting to set name: {name}") # Debug print
    state = await ctx.get("state")
    if state is None: # Initialize state if it's None
        state = {}
    state["name"] = name
    await ctx.set("state", state)
    print(f"State after setting name: {state}") # Debug print
    return f"Name set to {name}"

async def main():
    stateful_workflow = AgentWorkflow.from_tools_or_functions(
        [set_name],
        llm=llm,
        system_prompt="You are a helpful assistant that can set a name based on user input. When the user tells you their name, use the 'set_name' function.",
        initial_state={"name": "unset"},
        verbose=True # Enable verbose logging for the workflow
    )

    stateful_workflow_context = Context(workflow=stateful_workflow)

 
    response = await stateful_workflow.run(
        user_msg="My name is Yogesh", 
        ctx=stateful_workflow_context
    )
    print(f"Workflow response: {str(response)}")

    # Retrieve and print the state to verify
    final_state = await stateful_workflow_context.get("state")
    if final_state and "name" in final_state:
        print(f"Name as stored in final state: {final_state['name']}")
    else:
        print("Name not found in final state.")


if __name__ == "__main__":
    asyncio.run(main())
    