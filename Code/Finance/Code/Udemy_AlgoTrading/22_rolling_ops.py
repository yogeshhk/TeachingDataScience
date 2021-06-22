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

close_prices.fillna(method='bfill',axis=0,inplace=True) #replace NaN values using backfill method
close_prices.dropna(axis=0,inplace=True) #drop row containing any NaN value

daily_return = close_prices.pct_change() # Creates dataframe with daily return for each stock" are missing

# Rolling mean and standard deviation
daily_return.rolling(window=20).mean() # simple moving average
daily_return.rolling(window=20).std()

daily_return.ewm(span=20,min_periods=20).mean() # exponential moving average
daily_return.ewm(span=20,min_periods=20).std()




