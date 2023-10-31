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

#%%231031

def cluster_(df, target_name, pca=True):
    if pca:
        component = int(input('n_components를 입력하세요: '))
        pca_ = PCA(n_components=component)
        pca_transformed = pca_.fit_transform(df.iloc[:, :-1])
        df['pca_x'] = pca_transformed[:, 0]
        df['pca_y'] = pca_transformed[:, 1]
        
        kmeans = KMeans(n_clusters=len(df[target_name].unique()), init='k-means++', max_iter=500, random_state=0)
        kmeans.fit(df.loc[:, ['pca_x', 'pca_y']])
        df['pca_cluster'] = kmeans.labels_
        print(df.groupby(['target', 'pca_cluster']))
        markers=['s', 'x', 'o']
        for i in range(len(df[target_name].unique())):
            plt.scatter(df[df.pca_cluster==i].pca_x, df[df.pca_cluster==i].pca_y, marker=markers[i])
        plt.title('PCA_CLUSTER')
        return plt
    else:
        kmeans = KMeans(n_clusters=len(df[target_name].unique()), init='k-means++', max_iter=500, random_state=0)
        kmeans.fit(df.iloc[:, :-1])
        df['cluster'] = kmeans.labels_
        print(df.groupby(['target', 'cluster']).count())
        markers=['s', 'x', 'o']
        for i in range(len(df[target_name].unique())):
            plt.scatter(df[df.cluster==i].pca_x, df[df.cluster==i].pca_y, marker=markers[i])
        plt.title('CLUSTER')
        return plt

def cust_cluster(df):
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=1000, random_state=0)
    labels = kmeans.fit_predict(df.values)