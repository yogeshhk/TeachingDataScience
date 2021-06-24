# =============================================================================
# Import OHLCV data and transform it to Renko
# Author : Mayank Rasu (http://rasuquant.com/wp/)

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import yfinance as yf
import datetime as dt
from stocktrends import Renko

################################PLEASE READ ME####################################
#Stocktrends' author has renamed get_bricks() function to get_ohlc_data()
#therefore you may get error when trying to run line 43 below
#if that is the case please comment out line 43 and remove # sign from line 44 and rerun
##################################################################################

# Download historical data for required stocks
ticker = "AAPL"
ohlcv = yf.download(ticker,dt.date.today()-dt.timedelta(1825),dt.datetime.today())

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.loc[:,["Date","Open","High","Low","Adj Close","Volume"]]
    df.rename(columns = {"Date" : "date", "High" : "high","Low" : "low", "Open" : "open","Adj Close" : "close", "Volume" : "volume"}, inplace = True)
    df2 = Renko(df)
    df2.brick_size = round(ATR(DF,120)["ATR"][-1],0)
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df


renko_data = renko_DF(ohlcv)
renko_data.dropna(inplace=True)
print(renko_data.head())