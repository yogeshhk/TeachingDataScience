from adk import Agent
from adk.models import GeminiModel
import yfinance as yf

def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a given symbol."""
    # Implementation using yfinance or similar
    stock = yf.Ticker(symbol)
    return {"symbol": symbol, "price": stock.info.get('currentPrice')}

agent = Agent(
    model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[get_stock_price],
    instructions="You are a helpful financial assistant. Use tools when needed."
)

response = agent.run("What is NVIDIA's current stock price?")
print(response.text)