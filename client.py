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
#231101

text_sample ='''When I choose to see the good side of things, I'm not being naive. It is strategic and necessary. It’s how I’ve learned to survive through everything.'''
'''
nltk.download('stopwords')
#nltk.download('all')
stopwords_list = nltk.corpus.stopwords.words('english')
re_text_sample = pm.token_text(text_sample, stopwords_list)
print(re_text_sample)


file = os.path.dirname(os.path.dirname(__file__)) + '\\class file'
review_df = pd.read_csv(file + '\\labeledTrainData.tsv', header=0, sep='\t', quoting=3)
print(review_df.head())

data = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])
col_pos = np.array([2, 5, 0 ,1 , 3, 4, 5, 1, 3, 0, 3, 5, 0])
row_pos_idx = np.array([0, 2, 7, 9, 10, 12, 13])
print(sparse.csr_matrix((data, col_pos, row_pos_idx)).toarray())
'''
news = fetch_20newsgroups()
print(news.keys())

