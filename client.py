import python_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold
from urllib import request

#%%231018

path = os.path.dirname(__file__)
parent = os.path.dirname(path)
folder = parent + '\\class file'

'''
df = pd.read_csv(folder + '\\heart_disease.csv')
print(df.head())
df.info()
print(df.isna().sum())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

kfold=StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
model = XGBClassifier()
params = {'learning_rate':[0.001, 0.05, 0.1], 'max_depth':[2, 3], 'gamma':[0.001, 0.01, 0.1], 'min_child_weight':[1, 2, 3], 'subsample':[0.5, 0.7, 0.8], 'colsample_bytree':[0.5, 0.7, 0.8, 1]}
pm.grid_search(params, X, y, model, kfold, 'accuracy')
'''

df = pd.read_csv(folder + '\\heart_disease.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
#model=XGBClassifier()
#print(pm.early_end(X, y, model, 0))
#print(pm.early_end(X, y, model, 10))
#model=XGBClassifier(n_estimators=5000)
#print(pm.early_end(X, y, model, 100))

#pm.grid_search({'n_estimators':[25, 50, 75, 100, 200, 300, 400, 500]}, X, y)
#pm.grid_search({'n_estimators':[400], 'max_depth':[2, 3, 4, 5, 6, 7, 8, 9, 10]}, X, y)
#pm.grid_search({'n_estimators':[400], 'max_depth':[7], 'learning_rate':[0.01, 0.05, 0.075, 0.08, 0.1, 0.2, 0.3]}, X, y)
#pm.grid_search({'n_estimators':[400], 'max_depth':[7], 'learning_rate':[0.08], 'min_child_weight':[1, 2, 3, 4, 5}, X, y)
#pm.grid_search({'n_estimators':[400], 'max_depth':[7], 'learning_rate':[0.08], 'min_child_weight':[1, 2, 3, 4, 5]}, X, y)
#pm.grid_search({'n_estimators':[400], 'max_depth':[7], 'learning_rate':[0.08], 'min_child_weight':[5], 'subsample':[0.5, 0.6, 0.7, 0.8, 0.9]}, X, y)
#best_params : {'learning_rate': 0.08, 'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 400, 'subsample': 0.6}
'''
df = pd.read_csv(folder + '\\atlas-higgs-challenge-2014-v2.csv.gz', nrows=250000, compression='gzip')
df['Label'].replace(('s', 'b'), (1, 0), inplace=True)
#print(df.head())
df.drop(['KaggleSet', 'Weight'], axis=1, inplace=True)
df.rename(columns={'KaggleWeight':'Weight'}, inplace=True)
X=df.loc[:, ~df.columns.isin(['EventId', 'Label', 'Weight'])]
y = df.loc[:, 'Label']

print(df.Label.value_counts(normalize=True))
df['test_Weight'] = df['Weight']*550000 / len(y)
#params = {'n_estimators':120, 'learning_rate':0.1, 'missing':-999.0}
pm.atlas_(df)
'''
#url = 'https://raw.githubusercontent.com/rickiepark/handson-gb/main/Chapter09/cab_rides.csv'
#filename = 'cab_rides.csv'
#request.urlretrieve(url, filename)
#filename = 'weather.csv'
#request.urlretrieve(url, filename)

df = pd.read_csv(folder+'\\weather.csv')
#print(df.head())
#df.info()
#print(df[df.isna().any(axis=1)])
df.dropna(inplace=True)
#print(df[df.isna().any(axis=1)])
df['date'] = pd.to_datetime(df.time_stamp, unit='ms')
#print(df.date)
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour
df['dayofweek'] = df.date.dt.dayofweek

#print(df.head())
df['weekend'] = df['dayofweek'].apply(pm.to_week)
#print(df.head())
df['rush_hour'] = df.apply(pm.rush_hour, args=('hour', 'weekend'), axis=1)
#print(df.head())
#print(df.cab_type.value_counts())
df.info()
