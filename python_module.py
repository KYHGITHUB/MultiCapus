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

#%%231020

def svc_plot(X, y):
    clf = SVC(kernel = 'linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    a= -w[0]/w[1]
    b = -b/w[1]
    half_margin = 1/w[1]
    x_data = np.arange(X.max()+2)
    print(x_data)
    hyperline = a*x_data + b
    up_line = hyperline + half_margin
    down_line = hyperline - half_margin

    plt.figure(figsize=(6,4))
    plt.plot(X[:, 0][y==y[0]], X[:, 1][y==y[0]], 'rx')
    plt.plot(X[:, 0][y!=y[0]], X[:, 1][y!=y[0]], 'bo')
    plt.plot(x_data, hyperline)
    plt.plot(x_data, up_line, '--', c='green')
    plt.plot(x_data, down_line, '--', c='green')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.show()

#def cancer_svc_plot(X, y):
    