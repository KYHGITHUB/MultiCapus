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


#%%231115

def make_GRU(train_data, train_target, val_data, val_target,num_words=500, acti=None, optimizer=None, loss=None, metrics=None, checkpoint=None, earlystopping=None, epoch=100, batch_size=32):
    model4 = keras.Sequential()
    if keras.utils.to_categorical(train_data).shape[-1] != num_words:
        return print('warning!! : please enter the num_words value accurately')
    model4.add(keras.layers.Embedding(num_words, 16, input_length=train_data.shape[-1]))
    model4.add(keras.layers.GRU(8))
    model4.add(keras.layers.Dense(1, activation=acti))
    
    opt = optimizer
    model4.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    callback_list = []
    if checkpoint:
        callback_list.append(keras.callbacks.ModelCheckpoint('best_gru-model.h5', save_best_only=True))
    if earlystopping:
        callback_list.append(keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))
    history = model4.fit(train_data, train_target, epochs=epoch, batch_size=batch_size,
                     validation_data=(val_data, val_target),
                     callbacks=callback_list)
    return model4, history