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
#%%231102

file = os.path.dirname(os.path.dirname(__file__)) + '\\class file'
file_list = os.listdir(file)
for i in file_list:
    if i.startswith('top'):
        file_path = i
file2 = file +'\\'+file_path
file2_list = os.listdir(file2)
for i in file2_list:
    last_folder = i
last_path = file + '\\' + file_path + '\\' + last_folder
files = glob.glob(last_path + '\\*.data')
file_name = [file.split('\\')[5].split('.')[0] for file in files]
doc_list = []
for file in files:
    with open(file, 'r', encoding='latin-1') as f:
        doc_list.append(f.read())
document_df = pd.DataFrame({'filename':file_name, 'opinion_text':doc_list})
document_df_ = pm.cluster_text(document_df)
print(document_df_)