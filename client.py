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

#%%231026

file = os.path.dirname(os.path.dirname(__file__)) + '\\class file'
card = pd.read_csv(file + '\\creditcard.csv')
#card.info()
card.drop('Time', axis=1, inplace=True)

pm.case_get(card)