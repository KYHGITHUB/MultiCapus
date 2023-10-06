import python_module as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

#%%231006
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
rf_clf_acc = pm.RFC(iris_df, 'label')
print(f'랜덤 포레스트 예측 적중률 : {rf_clf_acc}')

kf_acc = pm.KFd(iris_df, 'label', n_split=5)
print(f'KFold 예측 적중률 : {kf_acc}')

skf_acc = pm.SKF(iris_df, 'label', n_split=3)
print(f'StratifiedKFold 예측 적중률 : {skf_acc}')

scores = pm.CVS(iris_df, 'label', 5)
print(f'cross_val_score 점수 : {scores}')
print(f'cross_val_score 평균 점수 : {np.mean(scores)}')

df = pm.GS(iris_df, 'label', 0.2, 3, [1,2,3,4,5,6,7,8,9,10], [1,2,3], 10)
print('GridSearchCV 결과')
print(df)