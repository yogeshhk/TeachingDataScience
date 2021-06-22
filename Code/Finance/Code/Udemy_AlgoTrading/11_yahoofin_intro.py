# -*- coding: utf-8 -*-
"""
yahoofin library intro

@author: Mayank Rasu (http://rasuquant.com/wp/)
"""

from yahoofinancials import YahooFinancials

ticker = 'MSFT'
yahoo_financials = YahooFinancials(ticker)
data = yahoo_financials.get_historical_price_data("2018-04-24", "2020-04-24", "daily")

print(data)
