import python_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

path = os.path.dirname(__file__)
parent = os.path.dirname(path)
file_path = parent + '\\class file'
df_bikes = pd.read_csv(file_path + '\\bike_rentals.csv')

#df_bikes.info()
df = pm.repair(df_bikes)
#print(df)
#df.to_csv('bike_cleaned.csv', index=False)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
'''
rmse = pm.get_lr(X, y)
print(f'LinearRegression : {rmse}')

rmse = pm.get_xg(X, y)
print(f'XGBRegressor : {rmse}')

rmse = pm.cross_score(LinearRegression(), X, y)
print(f'LinearRegression - cross_val_score : {rmse.mean()}')

rmse = pm.cross_score(XGBRegressor(), X, y)
print(f'XGBRegressor - cross_val_score : {rmse.mean()}')

rmse = pm.gbm_test(X, y)
print(f'gbm - DecisionTreeRegressor : {rmse}')
'''
learning_rate_values = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
rmse_dict={}
for value in learning_rate_values:
    rmse = pm.GBM(X, y, value)   
    rmse_dict[rmse] = value
best_learning_rate = rmse_dict[min(rmse_dict)]
print(f'min_rmse : {min(rmse_dict)}\nlearning_rate : {best_learning_rate}')


depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rmse_dict = pm.GBM_depth(X, y, depths)
best_depth = rmse_dict[min(rmse_dict)]
print(f'min_rmse : {min(rmse_dict)}\ndepth : {best_depth}')
#pm.plot_gbm_rmse(rmse_list, depths, 'depths')

