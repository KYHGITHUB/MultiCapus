# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
from prophet import Prophet
from datetime import datetime

def findCodes(name):
    code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download')
    code_df = code_all[0]
    expr_str = f'''회사명 == '{name}' '''
    code = code_df.query(expr_str)
    target_code = '{:06d}'.format(code['종목코드'].values[0])
    return target_code
    
def devTrainTest(df, column, date):
    df[column] = df[column].astype(int)
    train_df = df.query(f'''index <= '{date}' ''')
    test_df = df.query(f'''index > '{date}' ''')
    return train_df, test_df

def prePlot(df, column, date, degrees=10):
    train_df, test_df = devTrainTest(df, column, date)
    
    polynomial_features = PF(degree=degrees, include_bias=False)
    linear_regression = LR()
    pipeline = Pipeline( [ ('polynomial_features', polynomial_features), ('linear_regression', linear_regression) ] )
    x = np.arange(len(train_df.index)).reshape(-1,1)
    y = train_df[column]
    pipeline.fit(x, y)
   
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(x, pipeline.predict(x), label='Model', color='red')
    plt.scatter(x, y, edgecolors='b', s=5, label='Samples', color='skyblue')
    plt.title(f'Degree=degrees')
    plt.legend(loc='best')
    plt.show()

def prophetPlot(df, column, date, period):
    train_df, test_df = devTrainTest(df, column, date)
    
    df_prophet = train_df[column]
    df2 = df_prophet.reset_index()
    df2.columns = ['ds', 'y']
    m = Prophet()
    m.fit(df2)
    future = m.make_future_dataframe(periods=period)
    forcast = m.predict(future)
    m.plot(forcast)
    plt.show()
    
def toDateTime(val):
    return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

def timeDfAgg(df, column1, column2, *args):
    if type(df[column1][0]) != 'pandas._libs.tslibs.timestamps.Timestamp':
        if df[column2].isna().any():
            df.dropna(inplace=True)           
        df[column1] = df[column1].apply(toDateTime)
        df.set_index(column1, inplace=True)
        df_re = df.resample('d')
        result_df = df_re.agg(args)
    else:
        if df[column2].isna().any():
            df.dropna(inpace=True)
        df_re = df.resample('d')
        result_df = df_re.agg(args)

    return result_df