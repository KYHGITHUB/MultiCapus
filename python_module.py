# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#%%231024

def get_best_degree(model, degrees, X, y):
    result = {}
    for deg in degrees:
        pipe = Pipeline([(PolynomialFeatures().__class__.__name__, PolynomialFeatures(degree=deg, include_bias=False)), (model.__class__.__name__, model)])
        scores = cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=10)
        pipe.fit(X, y)
        result[deg] = (pipe, -scores.mean(), scores.std())
    
    mses = {}
    for i in result:
        #w = result[i][0].named_steps[model.__class__.__name__].coef_
        mses[i] = result[i][1]

    best_degree = min(mses, key=mses.get)

    X_test = np.linspace(0, 1, 100)
    y_test = true_func(X_test)
    X_test = X_test.reshape(-1, 1)
    plt.figure(figsize=(14,5))
    num=0
    for i in result:
        pipelines = result[i][0]
        mean_score = result[i][1]
        std_score = result[i][2]
        
        pred = pipelines.predict(X_test)
        num+=1
        ax = plt.subplot(1, len(degrees), num)
        plt.plot(X_test, pred, label = 'model')
        plt.plot(X, y, 'ko', label='samples')
        plt.plot(X_test, y_test, label = 'True function')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y', rotation=0)
        plt.title(f'Degree : {i}\nMSE : {mean_score}')
        plt.setp(ax, xticks=(), yticks=())
    plt.show()

    return best_degree
        
def true_func(X):
    return np.cos(1.5 * np.pi * X)
