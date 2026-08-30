from adk import Agent, Session
from adk.models import GeminiModel
from adk.storage import InMemoryStorage
import yfinance as yf

def get_stock_price(symbol: str) -> float:
    """Get current stock price."""
    return yf.Ticker(symbol).info.get('currentPrice', 0)

def get_company_info(symbol: str) -> dict:
    """Get company information."""
    ticker = yf.Ticker(symbol)
    return {"name": ticker.info.get('longName'),
            "sector": ticker.info.get('sector'),
            "summary": ticker.info.get('longBusinessSummary')}
    
# Create storage for session history
storage = InMemoryStorage()

agent = Agent(
    model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[get_stock_price, get_company_info],
    instructions="You are a financial advisor. Remember previous conversations."
)

# Create a session for this conversation
session = Session(
    agent=agent,
    storage=storage,
    session_id="user-123"
)

# Multi-turn conversation
response1 = session.run("What's NVIDIA's stock price?")
print(response1.text)

# Agent remembers previous context
response2 = session.run("How about their competitor AMD?")
print(response2.text)

# Session history is maintained
response3 = session.run("Compare both companies")
print(response3.text)