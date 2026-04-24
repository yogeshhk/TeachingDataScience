from adk import Agent
from adk.models import GeminiModel
from adk.extensions import GoogleSearchGrounding

agent = Agent(
    model=GeminiModel(
        model_name="gemini-2.0-flash-exp",
        extensions=[GoogleSearchGrounding()]
    ),
    instructions="Provide well-researched answers with citations."
)

response = agent.run(
    "What are the latest developments in AI semiconductor technology?"
)
print(response.text)
# Response will include citations from search results