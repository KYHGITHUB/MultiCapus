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

#%%231024

data = {'x' : [13, 19, 16, 14, 15, 14],
        'y' : [40, 83, 62, 48, 58, 43]}

X = pd.DataFrame(data['x'])
lr = LinearRegression()
lr.fit(X, data['y'])
pred = lr.predict(X)
#plt.plot(data['x'], data['y'], 'bo')
#plt.plot(X, pred, 'k--')
#plt.show()

#residuals = data['y'] - pred
#print((residuals**2).sum() / len(X))

#print(f"R2_score : {lr.score(X, data['y'])}")
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = pm.true_func(X) + np.random.randn(n_samples)*0.1
X = X.reshape(-1, 1)

degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_degree = pm.get_best_degree(LinearRegression(), degrees, X, y)
#print(f'best_degree : {best_degree}')