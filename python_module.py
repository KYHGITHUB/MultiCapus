# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score # 모델 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

#%% 231018

def grid_search(params, X, y, model=XGBClassifier(), cv=5, scoring='accuracy', random=False):
    clf = model
    if random:
        grid = RandomizedSearchCV(clf, params, cv=cv, n_jobs=-1, scroring=scoring, random_state=2)
    else:
        grid = GridSearchCV(clf, params, cv=cv, n_jobs=-1, scoring=scoring)
    grid.fit(X, y)
    print(f'best_params : {grid.best_params_}')
    print(f'best_score : {grid.best_score_}')

def early_end(X, y, model=XGBClassifier(), rounds=10, vb=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    clf = model
    clf.fit(X_train, y_train, eval_set= [(X_test, y_test)], eval_metric='error', early_stopping_rounds=rounds, verbose=vb)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)

def atlas_(df):
    X=df.loc[:, ~df.columns.isin(['EventId', 'Label', 'Weight'])]
    y = df.loc[:, 'Label']
    s = np.sum(df[df.Label==1])['test_Weight']
    b = np.sum(df[df.Label==0])['test_Weight']
    clf = XGBClassifier(n_estimators=120, learning_rate=0.1, missing=-999.0, scale_pos_weight=b/s)
    clf.fit(X, y, sample_weight=df['test_Weight'], eval_set=[(X, y)], eval_metric=['auc', 'ams@0.15'], sample_weight_eval_set=[df['test_Weight']])
    print(clf.evals_result())

def to_week(val):
    if val in [5, 6]:
        return 1
    else:
        return 0

def rush_hour(row, column1, column2):
    if (row[column1] in [6, 7, 8, 9, 15, 16, 17, 18]) and (row[column2] == 0):
        return 1
    else:
        return 0