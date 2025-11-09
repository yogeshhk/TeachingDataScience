from adk import Agent
from adk.models import GeminiModel
from pydantic import BaseModel, Field

def get_stock_price(symbol: str) -> float:
    """Get current stock price."""
    return yf.Ticker(symbol).info.get('currentPrice', 0)

def get_company_info(symbol: str) -> dict:
    """Get company information."""
    ticker = yf.Ticker(symbol)
    return {"name": ticker.info.get('longName'),
            "sector": ticker.info.get('sector'),
            "summary": ticker.info.get('longBusinessSummary')}
			
   
class StockAnalysis(BaseModel):
    symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price")
    recommendation: str = Field(description="Buy/Hold/Sell recommendation")
    reasoning: str = Field(description="Analysis reasoning")

agent = Agent(
    model=GeminiModel(model_name="gemini-2.0-flash-exp"),
    tools=[get_stock_price, get_company_info],
    output_schema=StockAnalysis,
    instructions="Analyze the stock and provide structured recommendation."
)

result: StockAnalysis = agent.run("Analyze NVIDIA stock")
print(f"Symbol: {result.symbol}")
print(f"Price: ${result.current_price}")
print(f"Recommendation: {result.recommendation}")
print(f"Reasoning: {result.reasoning}")