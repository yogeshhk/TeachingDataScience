# =============================================================================
# Import OHLCV data and perform basic data operations
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import datetime as dt
import yfinance as yf
import pandas as pd

# Download historical data for required stocks
tickers = ["MSFT","AMZN","AAPL","CSCO","IBM","FB"]

start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()
close_prices = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock

# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    close_prices[ticker] = yf.download(ticker,start,end)["Adj Close"]
    
# Mean, Median, Standard Deviation, daily return
close_prices.mean() # prints mean stock price for each stock
close_prices.median() # prints median stock price for each stock
close_prices.std() # prints standard deviation of stock price for each stock

print(close_prices)

daily_return = close_prices.pct_change() # Creates dataframe with daily return for each stock

daily_return.mean() # prints mean daily return for each stock
daily_return.std() # prints standard deviation of daily returns for each stock

print(daily_return)





