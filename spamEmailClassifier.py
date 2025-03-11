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





# Load the csv file into a dataframe
email_dataset = pd.read_csv("emails.csv")
email_dataset = email_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Stopword removal is the process of removing common words (like "is," "the," "and," "in") 
# from text because they do not add significant meaning.
# This helps in focusing on important words during text analysis.
nltk.download('punkt')
nltk.download('stopwords')

# separate the columns of csv file for spam classification
emails = email_dataset["text"]
spam_bits = email_dataset["spam"]
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


# Split data into training (60%), validation (20%), and test (20%)
X_train_text, X_test_val_text, Y_train, Y_test_val = train_test_split(
    filtering_of_all_emails, spam_bits, test_size=0.4, random_state=42,shuffle=True)
X_test_text, X_val_text, Y_test, Y_val = train_test_split(
    X_test_val_text, Y_test_val, test_size=0.5, random_state=42,shuffle=True)
# Apply TF-IDF vectorization only on the training data to prevent unnecessary data leakage
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)


# print(vectorizer.get_feature_names_out())

# print(vectorized_matrix.toarray())

# Now build a logistic regression model for the binary classification of spam vs non spam emails.
# First step : Split the dataset such that 60% used for training and 20% used for validation and
# hyperparameter tuning and 20% remaining for performance measure and final model evaluation(basically
# how accurate the model predicts for unforeseen data)


# Hyperparameter in logistic regression : C ( Regularization strength)
# Hyperparameter tuning usig GridSearchCV

c_grid = {'C':[0.001,0.01,0.1,1]}
# Now we want to test the regularization strengths and chose the best one out of it

grid_search = GridSearchCV(LogisticRegression(),c_grid,cv=5)
# Now we have to train it on training data which we have separated as 60% of dataset
grid_search.fit(X_train,Y_train)
# select the best model
best_model = grid_search.best_estimator_
best_hyperparam = grid_search.best_params_
print("Best hyperparameter : {}".format(best_hyperparam))
# Test our model using testing data
Y_val_predicted = best_model.predict(X_val)
testing_data_score = accuracy_score(Y_val,Y_val_predicted)
print("Accuracy score for testing data : {}".format(testing_data_score))
# Now apply this model to last 20% of our unforeseen data for final evaluation of the model

Y_test_predicted = best_model.predict(X_test)
validation_data_score = accuracy_score(Y_test , Y_test_predicted)
print("Accuracy score for final validation data : {}".format(validation_data_score))

# Checking any loopholes in my model as it is giving very high and unrealistic accuracy of 99.12% for testing data
# and 99.21% for validation data
# print(email_dataset["spam"].value_counts(normalize=True))  

# # Predict all emails as non-spam (majority class)
# majority_class_pred = [0] * len(Y_test)

# # Calculate accuracy
# baseline_accuracy = accuracy_score(Y_test, majority_class_pred)
# print("Baseline accuracy (predicting all non-spam):", baseline_accuracy)


# Still at 98% accuracy---> ????
# Trying to shuffle the datasep before processing for TF-IDF vectorization
# 97% accuracy -----> Somewhat better


print(classification_report(Y_test , Y_test_predicted))

joblib.dump(best_model, "spam_email_classification_model.pkl")
joblib.dump(vectorizer,"tf-idf-vectorizer.pkl")

