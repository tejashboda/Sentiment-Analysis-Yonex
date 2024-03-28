import numpy as np
import pandas as pd
import re
import emoji
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,f1_score
import streamlit as st
import pickle
result = None

st.title("Sentiment Analysis on User Reviews using ML")
text = st.text_input("Enter the Review")
import os

current_dir = os.path.dirname(__file__)

pickle_file_path = os.path.join(current_dir, "sentiment_yonex.pkl")

with open(pickle_file_path, "rb") as f:
    model = pickle.load(f)
if st.button("Submit")==True:
    result = model.predict([text])[0]

if result == 'Positive':
    st.caption(':green[Positive]')
    st.image('https://th.bing.com/th/id/OIP.Pgtw3L3HMTUcCP-RwVBMKwEfDZ?w=242&h=184&c=7&r=0&o=5&dpr=1.3&pid=1.7')
elif result == 'Negative':
    st.caption(':red[Negative]')
    st.image('https://th.bing.com/th/id/OIP.yvIS2S7zb6GSyV9SbSHpswHaHa?w=186&h=186&c=7&r=0&o=5&dpr=1.3&pid=1.7')