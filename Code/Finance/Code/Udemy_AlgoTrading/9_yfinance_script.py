# -*- coding: utf-8 -*-
"""
Getting data Using yfinance library

@author: Mayank Rasu (http://rasuquant.com/wp/)
"""

import yfinance as yf

# get ohlcv data for any ticker by period.
data = yf.download("MSFT", period='1mo', interval="5m")
print(data)

# get ohlcv data for any ticker by start date and end date
data = yf.download("MSFT", start="2017-01-01", end="2020-04-24")
print(data)

# get intraday data for any ticker by period.
data = yf.download("MSFT", period='1mo', interval="5m")
print(data)