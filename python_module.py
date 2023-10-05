# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # 모델 평가



#%%231005

def split_data(dict_, x_key, y_key, size=0.2, state=2):
    dict_copy = dict_.copy()
    x_train, x_test, y_train, y_test = train_test_split(dict_copy[x_key], dict_copy[y_key], test_size=size, random_state=state)
    return x_train, x_test, y_train, y_test

def DT(x_train, y_train, x_test, state=2):
    dt_clf = DecisionTreeClassifier(random_state=state)
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test) # 만들어진 모델에 x_test값 넣어서 예측값 도출하기
    return pred

def acc(y_test, pred):
    return accuracy_score(y_test, pred)
    