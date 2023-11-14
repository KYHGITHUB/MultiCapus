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
from tensorflow import keras


#%%231114

(train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)
train_input.shape, test_target.shape, train_target.shape, test_target.shape
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2,random_state=42)
train_seq = keras.preprocessing.sequence.pad_sequences(train_input, maxlen=100)
val_seq = keras.preprocessing.sequence.pad_sequences(val_input, maxlen=100)
train_oh = keras.utils.to_categorical(train_seq) # 원-핫 인코딩
val_oh = keras.utils.to_categorical(val_seq)

model, history = pm.make_RNN(train_oh, train_target, val_oh, val_target, last_floor='sigmoid', opt=keras.optimizers.RMSprop(learning_rate=1e-4), loss_='binary_crossentropy', metrics_='accuracy', checkpoint=True, epoch=100, batch=64)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.show()

#순환 신경망

