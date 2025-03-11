# Spam-Email-Classifier-Tinkerer-s-Lab-Project-


This repo contains source code of a ML model of Spam Email Classifier. This was done as a part of Tinkerer's Lab Club Project. 

I chose this project because it provides hands-on experience in ML, Natural Language Processing, and Data Preprocessing, which are essential for building real-world applications. I am particularly fascinated by how ML models can parse and understand human language to make intelligent decisions.
This project allows me to explore text classification, model training, and optimization, which are fundamental in spam detection, recommendation systems, and other AI-driven applications. By working on spam classification, I wanted to explore more in areas of NLP and classification systems, helping me build a strong foundation for future ML projects.

Problem Statement: 
Build a binary classifier to differentiate spam and non-spam emails.

Pre - requisites :
1) Python programming and usage of libraries like pandas for data preprocessing , nltk for NLP tasks and scikit-learn for building and training ML models (refer to documentations)
2) How do things like TF-IDF Vectorization works and Logistic regression concepts

Dataset :

https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data

Please look into the csv file in this repo emails.csv for the dataset I have been working on.

Resources and learning material are provided in this repo for the above project.

About Dataset
Dataset Name: Spam Email Dataset

Description:
This dataset contains a collection of email text messages, labeled as either spam or not spam. Each email message is associated with a binary label, where "1" indicates that the email is spam, and "0" indicates that it is not spam. The dataset is intended for use in training and evaluating spam email classification models.

Columns:

text (Text): This column contains the text content of the email messages. It includes the body of the emails along with any associated subject lines or headers.

spam_or_not (Binary): This column contains binary labels to indicate whether an email is spam or not. "1" represents spam, while "0" represents not spam.

Usage:
This dataset can be used for various Natural Language Processing (NLP) tasks, such as text classification and spam detection. Researchers and data scientists can train and evaluate machine learning models using this dataset to build effective spam email filters.


Installation and Setup Guide:

1) Firstly make sure that basic libraries like numpy , pandas and matplotlib are installed in the system. To install these (in Ubuntu terminal) ,
    sudo apt update
  sudo apt install python3-numpy python3-pandas python3-matplotlib


2) Next , we use a Python library for NLP called "nltk" (Natural Language ToolKit) . To install this in Ubuntu ,
   Open the terminal and run sudo apt update. Ensure that pip (Python package manager) is installed:sudo apt install python3-pip.
   Since nltk is available in Ubuntu’s package manager, you can install it using: sudo apt install python3-nltk

3) We use scikit-learn ML library. To set up this in Ubuntu terminal , use the folloing command.
    sudo apt install python3-sklearn


Trained Model (spamEmailClassifier.py) :
This model uses libraries like scikit-learn , nltk to train an ML model for spam classification of emails ( binary classifier)
. Firstly I have taken the entire dataset. Now I have split the dataset into 60% training data , 20% testing data and 20% final validation data.

Then , I have removed all the stopwords ( most common words that don't affect the way in which spam classification is done) from all the emails of training data and then fitted and vectorized that. Next , I trained a Logistic Regression model for training data and then tested my model on the testing data and validation data to find out the accuracy scores . Then I got a score of 99.12% and 99.21% on testing and validation data respectively.

Main Challenge I have faced in building this model : 
But this is highly unreal for a logistic regression model to be true to 99% above accuracy... Then I checked for any data leakage and working out on baseline accuracy to check whether the model is just menorizing or is it just putting up all zeroes due to inefficient shuffling of dataset... I also checked by lowering the hyperparameter.

By making up all the test cases and in my journey of finding out loopholes in my models and then fixing them , I have made it to 97% approx accuracy for testing and validation data which is somewhat better compared to previously unreal aaccuracies.

Key Learnings and Insights :
1) How to apply my previous knowledge of python and data preprocessing into real world project
2) hands on experience of working with ML models and gaining practical knowledge on many fundamentals like how to do hyperparameter tuning , how to resolve overfitting of models etc. I have found them very helpful as it helps me gain valuable experience on working with AI models and many bigger projects in future.
3) Clear Understanding of Logistic Regression and TF-IDF Vectorization and exploring on how are the things working around in TF-IDF like formulas to calculate the TF - IDF scores etc.
4) How to use this model for spam classification on real world datasets

Future interests and explorations:
I wanted to explore further into various ML models and algorithms that I can use for this spam email classfication in addition to this Logistic Regression like support vector machines and Deep Learning etc. 


Model Saving and Reuse :
I used joblib for saving and re-using the model for new dataset.
Model and Vectorizer are stored as .pkl files

Report on testing the model on a completely new dataset :

Dataset link : https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset

This was done using the file test_combined_data.py

It showed upto 74% accuracy .
This may also be one of the challenge I have faced in this project that the model which had performed extraordinarily well in one type of dataset with 97% accuracy score drastically dropped to 74% in the completely new dataset. So the possible reasons for this might be : 


1) Model may be overfit slightly to the previous dataset and therefore is less accurate to generalizations to new dataset
2) There may be some different spam patterns that the model is not aware of previously from training the previous dataset
3) Tf-IDF vectorization may have missed some new spam words.
4) May be model was trained with a limited spam variations that it could not grasp the vast differences easily.


So , I have created a new classifier model (train_and_test_combined_data.py) with same logistic regression but completely new one which is only trained to combined_data.csv file similar to previous model where it is trained with emails.csv file . Now this new model showed 98.5% accuracy approx for testing and validation data using 60-20-20 spliiting. This indicates that there is large variations in the vocabulary and structure of dataset and so the model should be exposed to large number of datasets to expand its knowledge and improve its accuracy score. 


Now with the file (test_combined_data_after_training) to add new vocabulary to the old model to expand the old model and it is working with 98.3% accuracy approx.
In this way , we can extend this to many datasets and continuously train and test new datasets. This is the best thing that I have learned in this project on training the AI models .



