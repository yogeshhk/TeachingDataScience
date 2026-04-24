#  https://docs.langchain.com/oss/python/langchain/overview

# pip install -qU langchain langchain-groq
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import ToolMessage

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Initialize Groq LLM (ensure GROQ_API_KEY is set in your environment)
llm = ChatGroq(model="llama-3.3-70b-versatile")  # https://console.groq.com/docs/models

agent = create_agent(
    model= llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant. When you call a tool, its output IS the final answer. Do not respond after tool output. Do not explain.",
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Pune"}]},
     return_intermediate_steps=True
)

tool_output = next(
    m.content for m in response["messages"] if isinstance(m, ToolMessage)
)

print(tool_output)