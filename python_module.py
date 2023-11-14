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



def make_RNN(train_data, train_target, val_data, val_target, last_floor, opt, loss_, metrics_, checkpoint=None, early_stopping=None, epoch=20, batch=32):
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(8, input_shape=(train_data.shape[1], train_data.shape[2])))
    model.add(keras.layers.Dense(1, activation=last_floor))
    
    model.compile(optimizer=opt, loss=loss_, metrics=metrics_)
    
    callback_list = []
    if checkpoint:
        checkpoint_cb = keras.callbacks.ModelCheckpoint('best_simpleRNN_model.h5', save_best_only=True)
        callback_list.append(checkpoint_cb)
    if early_stopping:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        callback_list.append(early_stopping)
    history = model.fit(train_data, train_target, epochs=epoch, batch_size=batch, validation_data=(val_data, val_target), callbacks=callback_list)
    
    return model, history