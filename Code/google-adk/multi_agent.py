from adk import Agent
from adk.models import GeminiModel

def web_search(query: str) -> str:
    """Search the web for information."""
    # Implementation using search API
    pass

def get_stock_data(symbol: str) -> dict:
    """Get comprehensive stock data."""
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    return {
        "price": ticker.info.get('currentPrice'),
        "recommendations": ticker.recommendations.tail(5).to_dict()
    }

web_agent = Agent(
    name="Web Agent",
    model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[web_search],
    instructions="Search the web and provide sourced information."
)

finance_agent = Agent(
    name="Finance Agent",
    model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[get_stock_data],
    instructions="Analyze financial data and present in clear tables."
)

from adk import Agent, MultiAgentOrchestrator
from adk.models import GeminiModel

orchestrator = MultiAgentOrchestrator(
    agents=[web_agent, finance_agent],
    coordinator=Agent(
        model=GeminiModel(model_name="gemini-2.0-flash-exp"),
        instructions="""You coordinate a team of specialized agents.
        - Use Web Agent for market news and trends
        - Use Finance Agent for stock data and analysis
        - Synthesize their outputs into comprehensive reports.""" ))

# Run multi-agent workflow
result = orchestrator.run(
    "Provide a comprehensive analysis of AI semiconductor companies "
    "including market outlook and financial performance")

print(result.text)

# Access individual agent outputs
for agent_name, agent_result in result.agent_outputs.items():
    print(f"\n{agent_name} output:")
    print(agent_result.text)