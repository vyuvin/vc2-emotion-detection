import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)
max_features = params['feature_engg']['max_features']

train_data = pd.read_csv("data/processed/train.csv").dropna(subset=['text'])
test_data = pd.read_csv("data/processed/test.csv").dropna(subset=['text'])
X_train = train_data['text'].values
y_train = train_data['sentiment'].values

X_test = test_data['text'].values
y_test = test_data['sentiment'].values
vectorizer = TfidfVectorizer(max_features=max_features)
# Fit the vectorizer on the training data and transform it to feature vectors
X_train_Tfidf = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer (do not fit again)
X_test_Tfidf = vectorizer.transform(X_test)
train_df = pd.DataFrame(X_train_Tfidf.toarray())
train_df['sentiment'] = y_train

test_df = pd.DataFrame(X_test_Tfidf.toarray())
test_df['sentiment'] = y_test

os.makedirs("data/interim", exist_ok=True)  # Ensure the directory exists
train_df.to_csv("data/interim/train_tfidf.csv", index=False)
test_df.to_csv("data/interim/test_tfidf.csv", index=False)
