# -*- coding: utf-8 -*-
"""
Vectorizing documents using NLTK

"""


text = "I am not a sentimental person but I believe in the utility of sentiment analysis"

# Tokenization
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print(tokens)

# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tokens=[lemmatizer.lemmatize(word) for word in tokens]

# Stemming
from nltk.stem import PorterStemmer
tokens=word_tokenize(text.lower())
ps = PorterStemmer()
tokens=[ps.stem(word) for word in tokens]
print(tokens)

# Stop words
import nltk
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

tokens_new = [j for j in tokens if j not in stopwords]