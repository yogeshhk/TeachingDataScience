"""
extracting data using alpha vantage

@author: Mayank Rasu (http://rasuquant.com/wp/)
"""

# importing libraries
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import os

# key_path = "D:\\Udemy\\Quantitative Investing Using Python\\1_Getting Data\\AlphaVantage\\key.txt"
# key=open(key_path,'r').read()
key = os.environ.get('ADVANTAGE_API_KEY',"")
if key == "":
    print("set the key first")
else:
    print(key)

# extracting data for a single ticker
ts = TimeSeries(key, output_format='pandas')
data = ts.get_daily(symbol='EURUSD', outputsize='full')[0]
data.columns = ["open","high","low","close","volume"]
data = data.iloc[::-1]
print(data)

# extracting stock data (historical close price) for multiple stocks
all_tickers = ["AAPL","MSFT","CSCO","AMZN","GOOG",
               "FB","BA","MMM","XOM","NKE","INTC"]
close_prices = pd.DataFrame()
api_call_count = 1
ts = TimeSeries(key=key, output_format='pandas')
print(ts)
start_time = time.time()
for ticker in all_tickers:
    try:
        data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='compact')[0]
        api_call_count+=1
        data.columns = ["open","high","low","close","volume"]
        data = data.iloc[::-1]
        close_prices[ticker] = data["close"]
    except Exception as e:
        print("Error ",e)
    if api_call_count==5:
        api_call_count = 1
        print("Need to sleep as cannot call > 5 APIs in a minute?")
        time.sleep(60 - ((time.time() - start_time) % 60.0))


# extracting ohlcv data for multiple stocks
all_tickers = ["AAPL","MSFT","CSCO","AMZN","GOOG",
               "FB","BA","MMM","XOM","NKE","INTC"]
ohlv_dict = {}
api_call_count = 1
ts = TimeSeries(key=key, output_format='pandas')
start_time = time.time()
for ticker in all_tickers:
    data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='compact')[0]
    api_call_count+=1
    data.columns = ["open","high","low","close","volume"]
    data = data.iloc[::-1]
    ohlv_dict[ticker] = data
    if api_call_count==5:
        api_call_count = 1
        time.sleep(60 - ((time.time() - start_time) % 60.0))