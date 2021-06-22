# =============================================================================
# Data extraction from stockrow.com
# Author : Reza Sadegehi (Reviewed and updated by Mayank Rasu)

# Please report bug/issues in the Q&A section
# =============================================================================
import pandas as pd 

from enum import Enum

tickers=["BA","AAPL"] #list of tickers whose data needs to be extracted
path = "D:\\Stockrow\\Financials" # please create this folder in your local machine
#Please also create subfolders Annual, Quartely and Trailing in your local machine
financials = {}
for ticker in tickers:
    financials[ticker] = {"Balance_Sheet":pd.DataFrame(),
                          "Income_Statement":pd.DataFrame(),
                          "CashFlow":pd.DataFrame()}

class Financials(Enum):
    
    Income_Statement = 1
    Balance_Sheet = 2
    CashFlow = 3
    Key_Metrics=4
    Growth=5

class Terms(Enum):
    Quarterly=1
    Annual=2
    Trailing=3

def FinFun(ticker,Fin,Term):
    """Please create subfolders Annual, Quartely and Trailing in your local machine"""
    link = "Income%20Statement"
    if Fin==Financials.Income_Statement:
        link = "Income%20Statement"
    elif Fin==Financials.Balance_Sheet:
        link = "Balance%20Sheet"
    elif Fin==Financials.CashFlow:
        link = "Cash%20Flow"
    elif Fin==Financials.Key_Metrics:
        link = "Metric"
    elif Fin==Financials.Growth:
        link = "Growth"
    
    if Term==Terms.Annual:
        URL="https://stockrow.com/api/companies/"+ticker+"/financials.xlsx?dimension=A&section="+link+"&sort=desc"
        
        filename= path+"\\Annual\\{}-Ann-{}.csv".format(link,ticker)
        return URL, filename
    
    elif Term==Terms.Quarterly:
        URL="https://stockrow.com/api/companies/"+ticker+"/financials.xlsx?dimension=Q&section="+link+"&sort=desc"
                 
        filename= path+"\\Quarterly\\{}-Qtr-{}.csv".format(link,ticker)
        return URL, filename   
    

    elif Term==Terms.Trailing:
        URL="https://stockrow.com/api/companies/"+ticker+"/financials.xlsx?dimension=T&section="+link+"&sort=desc"
     
        filename= path+"\\Trailing\\{}-ttm-{}.csv".format(link,ticker)
        return URL, filename   
        
               
def Download_data(ticker,Fin,Term):        
    
    global financials
    URL,filename= FinFun(ticker ,Fin,Term)
    print(" Download "+ str(Fin) +" for stock: " + ticker + " terms : " +str(Term) )

    df= pd.read_excel(URL)
    df.rename(columns={'Unnamed: 0':'Items'}, inplace=True)
    df.set_index("Items",inplace=True)
    df.columns = pd.to_datetime(df.columns).date
    df.to_csv(filename,index=True)
    financials[ticker][str(Fin).split(".")[-1]] = df
    


for ticker in tickers:
    Download_data(ticker ,Financials.Balance_Sheet,Terms.Annual)
    Download_data(ticker ,Financials.Income_Statement,Terms.Annual)
    Download_data(ticker ,Financials.CashFlow,Terms.Annual)

