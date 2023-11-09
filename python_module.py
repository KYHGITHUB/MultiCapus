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
#%%231109

def model_fn(X_data, y_target, optimizer_, epochs, dropout=None):
    train_scaled, val_scaled, train_target, val_target = train_test_split(X_data, y_target, test_size=0.2, random_state=42)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    opt = optimizer_
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='accuracy')

    history = model.fit(train_scaled, train_target, epochs=epochs, batch_size=128, validation_data=(val_scaled, val_target))

    return history, model