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
from sklearn.metrics import accuracy_score
import sys
from IPython.display import display
#%%231115
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

(train_input, train_target), (test_input, test_target) = keras.datasets.imdb.load_data(num_words=500)
train_input.shape, test_target.shape, train_target.shape, test_target.shape
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2,random_state=42)
train_seq = keras.preprocessing.sequence.pad_sequences(train_input, maxlen=100)
val_seq = keras.preprocessing.sequence.pad_sequences(val_input, maxlen=100)
train_oh = keras.utils.to_categorical(train_seq) # 원-핫 인코딩
val_oh = keras.utils.to_categorical(val_seq)
help(pm.make_GRU)
model, history = pm.make_GRU(train_seq, train_target, val_seq, val_target, num_words=500, acti='sigmoid',
                             optimizer=keras.optimizers.RMSprop(learning_rate=1e-4), loss='binary_crossentropy',
                            metrics='accuracy', checkpoint=True, earlystopping=True, epoch=100, batch_size=64)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.show()

model.evaluate(keras.preprocessing.sequence.pad_sequences(test_input, maxlen=100), test_target)
preds = model.predict(keras.preprocessing.sequence.pad_sequences(test_input, maxlen=100))

accuracy_score((preds>0.5).astype('int'), test_target)

#이미지 추천
path = os.path.abspath(sys.argv[0])
parent_path = os.path.dirname(path)
#parent_path = os.path.dirname(__file__)
class_path = parent_path + '\\class file'
class_path
files = os.listdir(class_path)
for i in files:
    if i.endswith('ids'):
        folder_name = i
images_folder_path = class_path + '\\' + folder_name
os.listdir(images_folder_path)
for i in files:
    if i.endswith('ids'):
        folder_name = i
images_folder_path = images_folder_path+ '\\' + folder_name
image_name_list = os.listdir(images_folder_path)
len(image_name_list)
images_folder_path
fashion_df = pd.read_csv(class_path + '\\fashion.csv')
fashion_df.head()
display(fashion_df)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
fashion_df.head()
fashion_df.info()

fashion_df.Category.value_counts()
fashion_df.Gender.value_counts()
fashion_men = fashion_df[fashion_df.Gender == 'Men']
