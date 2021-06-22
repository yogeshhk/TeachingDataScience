# -*- coding: utf-8 -*-
"""
handling NaN values

@author: Mayank Rasu (http://rasuquant.com/wp/)
"""

import datetime as dt
import yfinance as yf
import pandas as pd

stocks = ["AMZN","MSFT","FB","GOOG"]
start = dt.datetime.today()-dt.timedelta(3650)
end = dt.datetime.today()
cl_price = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock

# looping over tickers and creating a dataframe with close prices
for ticker in stocks:
    cl_price[ticker] = yf.download(ticker,start,end)["Adj Close"]
    
# filling NaN values
cl_price.fillna(method='bfill',axis=0,inplace=True)

#dropping NaN values
cl_price.dropna(axis=0,how='any')

print(cl_price.head())