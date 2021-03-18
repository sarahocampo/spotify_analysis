# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:50:19 2021

@author: sehoc
"""

import os
os.chdir(r'C:\Users\sehoc\OneDrive\Documents\GitHub\spotify_analysis')
import pandas as pd
import spot as sp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


data = pd.read_csv('data.csv')

x_data, y_data = sp.split_data(data) 
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

depthlist = [1, 2, 3, 4, 5,
             6, 7, 8, 9, 10,
             11, 12, 13]
for depth in depthlist:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(depth)
    print("Train Score: ", train_score)
    print("Test Score: ", test_score)
    

from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[3, 4, 5, 6, 7, 8, 9],
              'criterion' : ('gini', 'entropy'),
              'splitter' : ('best', 'random'),
              'max_features': ('auto', 'sqrt', 'log2')}
model = DecisionTreeClassifier()
clf = GridSearchCV(model, parameters)
clf.fit(x_train, y_train)
GridSearchCV(estimator=DecisionTreeClassifier(),
             param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
sorted(clf.cv_results_.keys())
