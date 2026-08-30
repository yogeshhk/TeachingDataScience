# https://console.groq.com/docs/agno

from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.tools.duckduckgo import DuckDuckGoTools

# Initialize the agent with an LLM via Groq and DuckDuckGoTools
agent = Agent(
    model=LMStudio(id="qwen/qwen3-1.7b"),
    description="You are an enthusiastic news reporter with a flair for storytelling!",
    tools=[DuckDuckGoTools()],      # Add DuckDuckGo tool to search the web
    show_tool_calls=True,           # Shows tool calls in the response, set to False to hide
    markdown=True                   # Format responses in markdown
)

# Prompt the agent to fetch a breaking news story from New York
agent.print_response("Tell me about a breaking news story from New York.", stream=True)
