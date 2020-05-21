# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:11:01 2020

@author: Ankush
"""


#Step 0 : Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Step 1: Import dataset
df_alexa = pd.read_csv('amazon_alexa.tsv', sep = '\t')
df_alexa.head(5)
df_alexa.keys()
df_alexa.tail(5)
df_alexa['verified_reviews']
df_alexa['variation']

#Step 2: Visualize the dataset
positive = df_alexa[df_alexa['feedback']==1]
negative = df_alexa[df_alexa['feedback']==0]

sns.countplot(df_alexa['feedback'], label = 'count')
sns.countplot(df_alexa['rating'], label = 'count')

df_alexa['rating'].hist(bins=5)

plt.figure(figsize=(40,15))
sns.barplot(x = 'variation', y = 'rating', data = df_alexa, palette = 'deep')


#Step 3: Data Cleaning
df_alexa = df_alexa.drop(['date','rating'], axis = 1)
df_alexa

variation_dummies = pd.get_dummies(df_alexa['variation'], drop_first = True)

df_alexa.drop(['variation'], axis = 1, inplace = True)

df_alexa = pd.concat([df_alexa,variation_dummies], axis = 1)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df_alexa['verified_reviews'])
alexa_countvectorizer.shape
print(vectorizer.get_feature_names())
print(alexa_countvectorizer.toarray())
 
df_alexa.drop(['verified_reviews'], axis = 1, inplace = True)

encoded_reviews = pd.DataFrame(alexa_countvectorizer.toarray())
df_alexa = pd.concat([df_alexa, encoded_reviews], axis = 1)
df_alexa

X = df_alexa.drop(['feedback'], axis=1)
X
y = df_alexa['feedback']
y

#Step 4: Training the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train, y_train)

#Step 5: Evaluation
y_predict= classifier.predict(X_test)
y_predict_train = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot = True)
print(classification_report(y_train, y_predict_train))

cm1 = confusion_matrix(y_test, y_predict)
sns.heatmap(cm1, annot = True)
print(classification_report(y_test, y_predict))

#Step 6: Improving the model 
df_alexa = pd.read_csv('amazon_alexa.tsv', sep = '\t')
df_alexa = pd.concat([df_alexa,pd.DataFrame(alexa_countvectorizer.toarray())], axis = 1)

df_alexa['length'] = df_alexa['verified_reviews'].apply(len)
df_alexa['length']

X = df_alexa.drop(['rating','date','variation','verified_reviews','feedback'], axis = 1)
y = df_alexa['feedback']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_predict= classifier.predict(X_test)   
cm3 = confusion_matrix(y_test, y_predict)
sns.heatmap(cm3, annot = True)
