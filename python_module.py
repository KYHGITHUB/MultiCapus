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

#%%231101
nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('all')
def token_text(text, stopwords_list):
    sentences = sent_tokenize(text)
    word_list = []
    remove_dict = {}
    for i in string.punctuation:
        remove_dict[ord(i)] = None
    for sentence in sentences:
        sentence = sentence.translate(remove_dict)
        words = word_tokenize(sentence)
        token_list = []
        for word in words:
            word = word.lower()
            if word not in stopwords_list and len(word) > 1:
                token_list.append(word)
        word_list.append(token_list)
    return word_list
    