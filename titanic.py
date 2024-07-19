#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:11:35 2024

@author: seanmilligan
"""

#importing packages
import numpy as np
import pandas as pd
import os

#check initial directory
print(os.getcwd())

desktop_path = os.path.expanduser("/Users/seanmilligan/Desktop/Python/Kaggle/Titanic")
os.chdir(desktop_path)

#check new directory
print(os.getcwd())

#loading in data
train_data = pd.read_csv("/Users/seanmilligan/Desktop/Python/Kaggle/Titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/Users/seanmilligan/Desktop/Python/Kaggle/Titanic/test.csv")
test_data.head()

#testing women v men fatality
women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women) / len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men) / len(men)

print('% of men who survived:', rate_men)

#random forest prediction
from sklearn.ensemble import RandomForestClassifier

#impute mean age in data
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)















