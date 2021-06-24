# =============================================================================
# Measuring the performance of a buy and hold strategy - CAGR
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import yfinance as yf
import numpy as np
import datetime as dt

# Download historical data for required stocks
ticker = "^GSPC"
SnP = yf.download(ticker,dt.date.today()-dt.timedelta(1825),dt.datetime.today())


def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    n = len(df)/252
    CAGR = (df["cum_return"][-1])**(1/n) - 1
    return CAGR

print(CAGR(SnP))