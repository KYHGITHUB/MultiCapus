# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
import matplotlib.style as style
style.use('ggplot')
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk import sent_tokenize, word_tokenize
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import keras



#%%231113

def make_conv2d(train_data, train_target, val_data, val_target, user_layer=None, optimizer_='adam', epoch=20):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=train_data.shape[1:]))
    model.add(keras.layers.MaxPooling2D(4))
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(7))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    if user_layer:
        model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    opt = optimizer_
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='accuracy')
    save_best_model = keras.callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(train_data, train_target, epochs=epoch, validation_data=(val_data, val_target), callbacks=[save_best_model, early_stopping])
    return model, history