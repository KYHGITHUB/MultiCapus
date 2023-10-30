# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
import matplotlib.style as style
style.use('ggplot')
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.decomposition import PCA

#%%231030

def get_model_down_dim(model, X_features, y_target):
    pca = PCA(n_components=2)
    pca.fit(X_features)
    X_features_pca = pca.transform(X_features)
    X_train, X_test, y_train, y_test = train_test_split(X_features_pca, y_target, random_state=2)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = model
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f'{clf.__class__.__name__} 의 정확도 : {accuracy_score(y_test, pred)}')
    return clf