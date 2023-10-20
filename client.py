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

#%%231020
'''
X = np.array([[3,4],[1,4],[2,3],[6,-1],[7,-1],[5,-3]] )
y = np.array([-1,-1, -1, 1, 1 , 1 ])

pm.svc_plot(X, y)
'''
cancer=load_breast_cancer()
print(cancer.data)