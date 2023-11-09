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
import datetime as dt
import nltk
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
import glob
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras

#%%231109
'''
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

#print(data.shape, labels.shape)
#print('=='*10, 'labels', '=='*10)
#print(labels)

model = Sequential()
model.add(Dense(32, activation='sigmoid', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.add(Dense(20, activation='softmax'))
model.add(Dense(5, activation='relu'))

#model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics='accuracy')
model.fit(data, labels, epochs=40, batch_size=214)

x_test = data[:50]
y_test = labels[:50]
print(x_test.shape, y_test.shape)
model.evaluate(x_test, y_test)
'''
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

history, model = pm.model_fn(train_scaled, train_target, optimizer_='adam', epochs=20)
print(model.evaluate(test_input/255.0, test_target))