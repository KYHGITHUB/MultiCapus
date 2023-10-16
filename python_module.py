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
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

#%% 231016

def repair(df):
    df['windspeed'].fillna((df['windspeed'].median()), inplace=True)
    df['hum'].fillna(df.groupby(['season'])['hum'].transform('median'), inplace=True)
    temp_mean = (df.iloc[700]['temp'] + df.iloc[702]['temp']) / 2
    atemp_mean = (df.iloc[700]['atemp'] + df.iloc[702]['atemp']) / 2
    df['temp'].fillna(temp_mean, inplace=True)
    df['atemp'].fillna(atemp_mean, inplace=True)
    df['dteday'] = df['dteday'].apply(pd.to_datetime, infer_datetime_format=True, errors='coerce')
    df['mnth'] = df['dteday'].dt.month
    df.loc[730, 'yr'] = 1.0
    df.drop(['casual', 'registered'], axis=1, inplace=True)
    df.drop(['instant', 'dteday'], axis=1, inplace=True)
    return df

def get_lr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return mse**0.5

def get_xg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    xg_reg = XGBRegressor()
    xg_reg.fit(X_train, y_train)
    pred = xg_reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return np.sqrt(mse)

def cross_score(model, X, y):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    return np.sqrt(-scores)

def gbm_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    tree1 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree1.fit(X_train, y_train)
    y_train_pred = tree1.predict(X_train)
    y2_train = y_train - y_train_pred

    tree2 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree2.fit(X_train, y2_train)
    y2_train_pred = tree2.predict(X_train)
    y3_train = y2_train - y2_train_pred

    tree3 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree3.fit(X_train, y3_train)
    
    y1_pred = tree1.predict(X_test)
    y2_pred = tree2.predict(X_test)
    y3_pred = tree3.predict(X_test)
    y_pred = y1_pred + y2_pred + y3_pred

    return np.sqrt(mean_squared_error(y_test, y_pred))

def GBM(X, y, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    gbr = GradientBoostingRegressor(max_depth=2, n_estimators=300, random_state=2, learning_rate=learning_rate)
    gbr.fit(X_train, y_train)
    pred = gbr.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))

def plot_gbm_rmse(rmse_list, x=None, values='learning_rate'):
    if x is None:
        x = np.arange(len(rmse_list))
    plt.plot(x, rmse_list)
    plt.xlabel(values)
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()

def GBM_depth(X, y, depths):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    rmse_dict = {}
    for depth in depths:
        gbr = GradientBoostingRegressor(max_depth=depth, n_estimators=300, random_state=156, learning_rate=0.2)
        gbr.fit(X_train, y_train)
        pred = gbr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_dict[rmse] = depth
    return rmse_dict