from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # Add a print here so you can see when the function actually runs!
    print(f"--- Executing get_weather for {city} ---")
    return f"It's always sunny in {city}!"

llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# In v1.0, create_agent creates a compiled graph.
# We need to make sure the agent is permitted to call tools.
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. If you need weather info, use the tool."
)

# Use the invoke method
inputs = {"messages": [("user", "What is the weather in Pune?")]}
response = agent.invoke(inputs)

# In v1.0, the response contains the full message history of the turn.
# The last message should now be the assistant's final answer AFTER the tool result.
for msg in response["messages"]:
    msg.pretty_print()