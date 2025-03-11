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



# Load the csv file into a dataframe
email_dataset = pd.read_csv("combined_data.csv")
email_dataset = email_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# separate the columns of csv file for spam classification
emails = email_dataset["text"]
spam_bits = email_dataset["label"]
filtering_of_all_emails = []
stop_words = set(stopwords.words("english"))

for email in emails :
    words_of_email = word_tokenize(email)
    
    filtered_words = []
    for word in words_of_email :
        if word.lower() not in stop_words :
            filtered_words.append(word)
        else :
            continue 
    filtering_of_all_emails.append(" ".join(filtered_words))

X_test = loaded_vectorizer.transform(filtering_of_all_emails)
Y_test = loaded_model.predict(X_test)

accuracy_of_loaded_model = accuracy_score(Y_test , spam_bits)
print("Accuracy of loaded model on the new dataset combined_data.csv is : {}".format(accuracy_of_loaded_model))


print(classification_report(Y_test, spam_bits))
