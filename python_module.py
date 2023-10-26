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

#%%231026

def get_frc(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print(f'TN, FP : {np.round(confusion, 4)[0]}')
    print(f'FN, TP : {np.round(confusion, 4)[1]}')
    print(f'accuracy : {accuracy}')
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'f1 : {f1}')
    print(f'roc_auc : {roc_auc}')

def get_pred_predproba(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1], test_size=0.2, random_state=156)
    logr = LogisticRegression()
    logr.fit(X_train, y_train)
    pred = logr.predict(X_test)
    pred_proba = logr.predict_proba(X_test)[:, 1]
    return y_test, pred, pred_proba

def case_get(df):
    cases = ['case1', 'case2', 'case3']
    for case_ in cases:
        df_copy = df.copy()
        if case_ == 'case1':
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount 변환 x ======')
            get_frc(y_test, pred, pred_proba)
        if case_ == 'case2':
            scaler = StandardScaler()
            Amount_scaled = scaler.fit_transform(np.array(df_copy.Amount).reshape(-1, 1))
            df_copy['Amount'] = Amount_scaled
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount use StandardScaler ======')
            get_frc(y_test, pred, pred_proba)
        if case_ == 'case3':
            Amount_scaled = np.log1p(np.array(df_copy.Amount))
            df_copy['Amount'] = Amount_scaled
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount use log1p ======')
            get_frc(y_test, pred, pred_proba)
