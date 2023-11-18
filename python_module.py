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

#%%231116
def cat_or_dog(train_generator, validation_generator, batch_size, optimizer, loss, metrics, epoch):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    step = train_generator.n / batch_size
    val_step = validation_generator.n / batch_size
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=step,
                                  epochs=epoch,
                                  validation_data=validation_generator,
                                  validation_steps=val_step)
    return model, history