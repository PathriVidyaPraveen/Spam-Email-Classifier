# Spam-Email-Classifier-Tinkerer-s-Lab-Project-


Disclaimer : Project currently under development.


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





