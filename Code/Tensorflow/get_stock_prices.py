# https://github.com/ranaroussi/yfinance
import yfinance as yf
import pandas as pd

def get_tickers(filename="data/symbols.txt"):
    tickers = []
    try:
        with open(filename,'r') as fin:
            raw_lines = fin.readlines()
            tickers = [v.strip() for v in raw_lines]
    except Exception as e:
        print("Cannot find {}".format(filename))
    return tickers

def get_prices(tickers):
    data_df = None
    try:
        list_of_dicts = []
        for stock in tickers:
            info = yf.Ticker(stock).info
            stock_dict = {'stock':stock,
                          'Open': info.get('regularMarketOpen'),
                          'Close':info.get('regularMarketPreviousClose'),
                          'Volume':info.get('regularMarketVolume'),
                          'High':info.get('regularMarketDayHigh'),
                          'Low':info.get('regularMarketDayLow')}
            list_of_dicts.append(stock_dict)
        data_df = pd.DataFrame(list_of_dicts)
    except Exception as e:
        print("Cannot find prices")
    return data_df


if __name__ == "__main__":
    tickers = get_tickers()
    data_df = get_prices(tickers)
    data_df.to_csv("prices.csv",index=False)
    print(data_df.head())