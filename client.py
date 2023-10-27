import python_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from urllib import request
from sklearn.datasets import load_iris, load_breast_cancer
import seaborn as sns

#%%231027

file = os.path.dirname(os.path.dirname(__file__)) + '\\class file'

house_df = pd.read_csv(file+'\\hpr_data.csv')
#house_df.info()
#print(house_df.isna().sum().sort_values(ascending=False)[:10])
#print(len(house_df))
#print(1460*0.2)
#print(house_df.isna().sum()[house_df.isna().sum()>1460*0.2])
del_list = list(house_df.isna().sum()[house_df.isna().sum()>1460*0.2].index)
#print(del_list)
house_df.drop(del_list, axis=1, inplace=True)
print(house_df.dtypes)
#house_df.fillna(
#nan_list = list(house_df.isna().sum().sort_values(ascending=False)[:13].index)
#print(house_df.dtypes[nan_list])
X = house_df.drop('PRICE', axis=1)
y = house_df.PRICE
y_log = np.log1p(y)

#sns.distplot(y)
#plt.show()
#sns.distplot(y_log)
#plt.show()
#X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=156)
