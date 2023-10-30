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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%231030

iris = load_iris()
X_features = iris.data
y_target = iris.target

model_ = RandomForestClassifier(random_state=42)
model = pm.get_model_down_dim(model_, X_features, y_target)

test_iris_idx = np.random.choice(len(iris.data), int(len(iris.data)*0.3))
test_iris_data = iris.data[test_iris_idx]
test_iris_target = iris.target[test_iris_idx]
test_iris_data = StandardScaler().fit_transform(test_iris_data)
test_iris_data = PCA(n_components=2).fit_transform(test_iris_data)


pred = model.predict(test_iris_data)
#print(accuracy_score(test_iris_target, pred))

plt.scatter(test_iris_data[:, 0], test_iris_data[:, 1], c=test_iris_target)
plt.scatter(test_iris_data[:, 0], test_iris_data[:, 1], c=pred, marker='x')
plt.show()