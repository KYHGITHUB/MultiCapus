# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import font_manager

def selectBorough(borough):
    if '서울' in borough:
        return borough.split(' ')[1][:-1]
    elif len(borough.split(' ')) == 1:
        return borough[:-1]
    else:
        return borough.split(' ')[0]
    
def setCCTV(df):
    df1 = df.groupby(df['관리기관명'], as_index=False).sum().loc[:,['관리기관명', '카메라대수']].copy()
    df1['관리기관명'] = df1['관리기관명'].apply(selectBorough)
    df_cctv = df1
    return df_cctv

def setPOP(df):
    df_pop = df.copy()
    df_pop.rename(columns={'동별(2)':'관리기관명', '2023 1/4.1':'총_인구수'}, inplace=True)
    df_pop = df_pop[['관리기관명','총_인구수']]
    df_pop.drop(index=[0,1], inplace=True)
    return df_pop

def mergedf(df1, df2):
    new_df = pd.merge(setCCTV(df1), setPOP(df2))
    new_df['총_인구수'].values
    new_df['총_인구수'] = new_df['총_인구수'].astype(int)
    return new_df

def LRdf(df1, df2):
    new_df = mergedf(df1, df2)
    x = new_df['총_인구수'].values.reshape(-1,1)
    y = new_df['카메라대수']
    model = LinearRegression()
    model.fit(x,y)
    new_df['predict_lr'] = model.predict(x)
    new_df['residual'] = new_df['카메라대수'] - new_df['predict_lr']
    new_df_sort = new_df.sort_values(by='residual', ascending=False)
    new_df_sort.reset_index(inplace=True)
    new_df_sort.drop(columns='index', inplace=True)
    return new_df_sort

def PredictModelPlot(df1, df2):
    new_df_sort = LRdf(df1, df2)
    
    font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'
    fontprop = font_manager.FontProperties(fname=font_path, size=20)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    scatter = ax.scatter(new_df_sort['총_인구수'], new_df_sort['카메라대수'], c=new_df_sort['residual'], label='자치구')
    ax.plot(new_df_sort['총_인구수'], new_df_sort['predict_lr'], c='r', label='예측값')
    ax.legend(prop=fontprop)
    ax.set_xlabel('총 인구수', fontproperties=fontprop)
    ax.set_ylabel('카메라대수', fontproperties=fontprop)
    plt.colorbar(scatter, ax=ax)
    for n in range(3):
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, '카메라대수']*1.01,
                new_df_sort.loc[n, '관리기관명'], fontproperties=fontprop, size=10)
    for n in range(21, 24):
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, '카메라대수']*1.01,
                new_df_sort.loc[n, '관리기관명'], fontproperties=fontprop, size=10)
    plt.show()