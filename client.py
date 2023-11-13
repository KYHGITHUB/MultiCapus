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

#%%231113

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_data, train_target), (test_data, test_target) = keras.datasets.fashion_mnist.load_data()
#print(train_data.shape, test_data.shape)
#print(train_data[0].shape)

#fig = plt.figure(figsize=(15, 10))
#for n in range(100):
#    ax = fig.add_subplot(10, 10, n+1)
#    ax.imshow(train_data[n])
#plt.show()
train_scaled = train_data.reshape(-1, 28, 28, 1) / 255.0
#print(train_scaled.shape)
train_scaled_data, val_data, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

model, history = pm.make_conv2d(train_scaled_data, train_target, val_data, val_target, user_layer=True)
print(model.evaluate(val_data, val_target))
preds = model.predict(test_data[:1])
plt.bar(range(1,11), preds[0])
preds[0]
plt.imshow(test_data[0])
