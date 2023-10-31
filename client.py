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
#%%231031
'''
iris = load_iris()
irisDF = pd.DataFrame(iris.data, columns=iris.feature_names)
irisDF['target'] = iris.target

ax = pm.cluster_(irisDF, 'target', pca=True)
ax.show()
ax = pm.cluster_(irisDF, 'target', pca=False)
ax.show()
'''
file = os.path.dirname(os.path.dirname(__file__)) + '\\class file'
retail_df = pd.read_excel(file+'\\Online_Retail.xlsx')
pd.set_option('display.max_columns', None)
#print(retail_df.head())
#retail_df.info()
#print(retail_df.Country.value_counts())
retail_df = retail_df[retail_df.Country=='United Kingdom']
retail_df = retail_df[retail_df.Quantity>0]
retail_df = retail_df[retail_df.UnitPrice>0]
retail_df = retail_df[retail_df.CustomerID.notnull()]
retail_df['sale_amount'] = retail_df.Quantity * retail_df.UnitPrice
retail_df['CustomerID'] = retail_df.CustomerID.astype(int)
#retail_df.info()
#print(retail_df.head())
cust_df = retail_df.groupby('CustomerID').agg({'InvoiceNo':'count', 'InvoiceDate':'max', 'sale_amount':'sum'})
cust_df.rename(columns={'InvoiceNo':'Frequency', 'InvoiceDate':'Recency', 'sale_acount':'Monetary'}, inplace=True)
cust_df['Recency'] = dt.datetime(2011, 12, 11) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x : x.days)
cust_df.info()