# ============================================================================
# getting fundamental data from financialmodelingprep.com
# Author - Mayank Rasu

# Please report bugs/issues in the Q&A section
# =============================================================================

import requests
import pandas as pd

link = "https://financialmodelingprep.com/api/v3"
api_key = "your_api_key"  # generate you free API key and paste it here
tickers = ["AXP"]


#list of tickers whose financial data needs to be extracted
financial_dir = {}

for ticker in tickers:
    try:
    #getting balance sheet data
        temp_dir = {}
        url = link+"/balance-sheet-statement/"+ticker+"?apikey={}".format(api_key)
        page = requests.get(url)
        fin_dir = page.json()
        for key,value in fin_dir[0].items():
            temp_dir[key] = value
    #getting income statement data
        url = link+"/income-statement/"+ticker+"?apikey={}".format(api_key)
        page = requests.get(url)
        fin_dir = page.json()
        for key,value in fin_dir[0].items():
            if key not in temp_dir.keys():
                temp_dir[key] = value
    #getting cashflow statement data
        url = link+"/cash-flow-statement/"+ticker+"?apikey={}".format(api_key)
        page = requests.get(url)
        fin_dir = page.json()
        for key,value in fin_dir[0].items():
            if key not in temp_dir.keys():
                temp_dir[key] = value
    #getting EV data
        url = link+"/enterprise-value/"+ticker+"?apikey={}".format(api_key)
        page = requests.get(url)
        fin_dir = page.json()
        for key,value in fin_dir["enterpriseValues"][0].items():
            if key not in temp_dir.keys():
                temp_dir[key] = value
    #getting key statistic data
        url = link+"/company-key-metrics/"+ticker+"?apikey={}".format(api_key)
        page = requests.get(url)
        fin_dir = page.json()
        for key,value in fin_dir["metrics"][0].items():
            if key not in temp_dir.keys():
                temp_dir[key] = value
        
    #combining all extracted information with the corresponding ticker
        financial_dir[ticker] = temp_dir
        
    except Exception as e:
        print(e)
        
#storing information in pandas dataframe
combined_financials = pd.DataFrame(financial_dir)
  