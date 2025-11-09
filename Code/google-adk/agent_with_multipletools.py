from adk import Agent
from adk.models import GeminiModel
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

def get_analyst_recommendations(symbol: str) -> str:
    """Get analyst recommendations."""
    return yf.Ticker(symbol).recommendations.to_string()

agent = Agent(model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[get_stock_price, get_company_info, get_analyst_recommendations],
    instructions="Provide detailed financial analysis using available tools.")

for chunk in agent.run_stream("Write a report on NVIDIA stock"):
    print(chunk.text, end="", flush=True)