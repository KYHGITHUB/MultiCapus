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

#%%231006

#랜덤포레스트
def RFC(df, target, size=0.2, state=2):	# target : 종속변수
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=size, random_state=state)
    rf_clf = RandomForestClassifier(random_state=state)
    rf_clf.fit(x_train, y_train)
    pred = rf_clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    return acc

#KFold
def KFd(df, target, n_split=3, state=2):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    kfold = KFold(n_splits=n_split)
    cv_acc = []
    df_clf = DecisionTreeClassifier(random_state=state)
    for train_index, test_index in kfold.split(x_):
        df_clf.fit(x_.iloc[train_index], y_.iloc[train_index])
        pred = df_clf.predict(x_.iloc[test_index])
        cv_acc.append(accuracy_score(y_.iloc[test_index], pred))
    return np.mean(cv_acc)

#StratifiedKFold
def SKF(df, target, n_split=3, state=2):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    cv_acc = []
    skf = StratifiedKFold(n_splits=n_split)
    df_clf = DecisionTreeClassifier(random_state=state)
    for train_index, test_index in skf.split(x_, y_):
        df_clf.fit(x_.iloc[train_index], y_.iloc[train_index])
        pred = df_clf.predict(x_.iloc[test_index])
        cv_acc.append(accuracy_score(y_.iloc[test_index], pred))
    return np.mean(cv_acc)

#cross_val_score
def CVS(df, target, cv=3):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    dt_clf = DecisionTreeClassifier()
    scores = cross_val_score(dt_clf, x_, y_, scoring='accuracy', cv=cv)
    return scores

#GridSearchCV
def GS(df, target, size=0.2, state=2, max_depths=[1, 2, 3], min_samples_splits=[2,3], cv=3):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=size, random_state=state)
    dt_clf = DecisionTreeClassifier()
    params={'max_depth':max_depths, 'min_samples_split':min_samples_splits}
    grid_dt_clf = GridSearchCV(dt_clf, param_grid=params, cv=cv, refit=True)
    grid_dt_clf.fit(x_train, y_train)
    new_df = pd.DataFrame(grid_dt_clf.cv_results_)
    return new_df[['params', 'mean_test_score', 'rank_test_score']]
