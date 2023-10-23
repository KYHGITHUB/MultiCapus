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
from sklearn.datasets import load_iris, load_breast_cancer

#%%231023

filepath = os.path.dirname(os.path.dirname(__file__)) + '\\class file'
df = pd.read_csv(filepath + '\\santander_train.csv', encoding='latin-1')
print(df.head())