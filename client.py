import python_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

#%% 231005
data = {'x':[13, 19, 16, 14, 15, 18],
        'y' : [40, 83, 62, 57, 58, 63]}

data = pd.DataFrame(data)

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df)
print('='*50)
x_train, x_test, y_train, y_test = pm.split_data(iris, 'data', 'target')
pred = pm.DT(x_train, y_train, x_test)
check_df = pd.DataFrame({'real_value':y_test, 'prediction_value':pred})
check_df['bools'] = check_df['real_value'] == check_df['prediction_value']
print(check_df)
acc = pm.acc(y_test, pred)
print(acc)
print(check_df[check_df['bools']==False])
print(check_df['bools'].value_counts())