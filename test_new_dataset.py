import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import sklearn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


loaded_model = joblib.load("spam_email_classification_model.pkl")
loaded_vectorizer = joblib.load("tf-idf-vectorizer.pkl")
print("Loaded the model and vectorizer successfully")

# Example new email ( change this mail with spam or not spam mails and check the model accuracy )
new_email = ["Team Meeting Scheduled for Tomorrow"]
  

# Convert text using the saved TF-IDF vectorizer
new_email_vectorized = loaded_vectorizer.transform(new_email)

# Predict spam or not
prediction = loaded_model.predict(new_email_vectorized)

if prediction[0] == 1:
    print("This is SPAM!")
else:
    print("This is NOT spam.")
