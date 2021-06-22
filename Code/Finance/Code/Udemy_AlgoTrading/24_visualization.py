# =============================================================================
# Import OHLCV data and perform basic visualizations
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt


# Download historical data for required stocks
tickers = ["MSFT","AMZN","AAPL","CSCO","IBM","FB"]

close_prices = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock
start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()


# looping over tickers and creating a dataframe with close prices
for ticker in tickers:
    close_prices[ticker] = yf.download(ticker,start,end)["Adj Close"]
    
# Handling NaN Values
close_prices.fillna(method='bfill',axis=0,inplace=True) # Replaces NaN values with the next valid value along the column
daily_return = close_prices.pct_change() # Creates dataframe with daily return for each stock

# Data vizualization
close_prices.plot() # Plot of all the stocks superimposed on the same chart

cp_standardized = (close_prices - close_prices.mean())/close_prices.std() # Standardization
cp_standardized.plot() # Plot of all the stocks standardized and superimposed on the same chart

close_prices.plot(subplots=True, layout = (3,2), title = "Tech Stock Price Evolution", grid =True) # Subplots of the stocks


# Pyplot demo
fig, ax = plt.subplots()
# plt.style.available
plt.style.use('ggplot')
ax.set(title="Daily return on tech stocks", xlabel="Tech Stocks", ylabel = "Daily Returns")
plt.bar(daily_return.columns,daily_return.mean())
plt.show()