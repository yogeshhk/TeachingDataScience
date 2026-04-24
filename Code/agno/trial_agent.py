# https://github.com/agno-agi/agno

from agno.agent import Agent
from agno.models.lmstudio import LMStudio


agent = Agent(
    model=LMStudio(id="qwen/qwen3-1.7b"),
    markdown=True,
    stream=True,
    show_tool_calls=True
)

agent.print_response("what is Python?", stream=True)