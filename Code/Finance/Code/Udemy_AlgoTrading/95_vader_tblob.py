# -*- coding: utf-8 -*-
"""
Demo of VADER and TextBlob sentiment analyser 

@author: Mayank Rasu
"""

################################Vader Demo#####################################

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
analyser.polarity_scores("This is a good course")
analyser.polarity_scores("This is an awesome course") # degree modifier
analyser.polarity_scores("The instructor is so cool")
analyser.polarity_scores("The instructor is so cool!!") # exclaimataion changes score
analyser.polarity_scores("The instructor is so COOL!!") # Capitalization changes score
analyser.polarity_scores("Machine learning makes me :)") #emoticons
analyser.polarity_scores("His antics had me ROFL")
analyser.polarity_scores("The movie SUX") #Slangs


################################Textblob Demo##################################

from textblob import TextBlob

TextBlob("His").sentiment
TextBlob("remarkable").sentiment
TextBlob("work").sentiment
TextBlob("ethic").sentiment
TextBlob("impressed").sentiment
TextBlob("me").sentiment
TextBlob("His remarkable work ethic impressed me").sentiment
