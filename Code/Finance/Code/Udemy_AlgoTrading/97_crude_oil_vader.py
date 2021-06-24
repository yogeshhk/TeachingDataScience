"""
Web scraping crude oil news and using VADER for sentiment analysis

@author: Mayank Rasu
"""

# import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Create lists to store scraped news urls, headlines and text
url_list = []
date_time = []
news_text = []
headlines = [] 

for i in range(1,3): #parameters of range function correspond to page numbers in the website with news listings
    #get the list of unique urls in the page
    url = 'https://oilprice.com/Energy/Crude-Oil/Page-{}.html'.format(i)
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")
    for links in soup.find_all('div', {'class': 'categoryArticle'}):
        for info in links.find_all('a'):
            if info.get('href') not in url_list:
                url_list.append(info.get('href'))

for www in url_list:
    #access each url
    headlines.append(www.split("/")[-1].replace('-',' '))
    request = requests.get(www)
    soup = BeautifulSoup(request.text, "html.parser")
    
    #store date and time of publication of the article
    for dates in soup.find_all('span', {'class': 'article_byline'}):
        date_time.append(dates.text.split('-')[-1])
    
    #store the text of the news
    temp = []
    for news in soup.find_all('p'):
            temp.append(news.text)
    
    #identify the last line of the news article
    for last_sentence in reversed(temp):
        if last_sentence.split(" ")[0]=="By" and last_sentence.split(" ")[-1]=="Oilprice.com":
            break
        elif last_sentence.split(" ")[0]=="By":
            break
    
    #prune non news related text from the scraped data to create the news text
    joined_text = ' '.join(temp[temp.index("More Info")+1:temp.index(last_sentence)])
    news_text.append(joined_text)


# save news text along with the news headline in a dataframe      
news_df = pd.DataFrame({ 'Date' : date_time,
                         'Headline': headlines,
                         'News': news_text,
                       })

# use VADER to perform sentiment analysis on stored news articles
analyser = SentimentIntensityAnalyzer()

def comp_score(text):
   return analyser.polarity_scores(text)["compound"]   
  
news_df["sentiment"] = news_df["News"].apply(comp_score)

