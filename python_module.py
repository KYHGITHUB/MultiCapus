# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import font_manager

def delSpace(val):
    return val.replace(' ', '')

def delComma(val):
    return val.replace(',', '')

def dfPop(df_pop):
    df_pop.drop(index=[0,1], inplace=True)      
    df_pop.rename(columns={'동별(2)':'구분', '2023 1/4.1':'총 인구수'}, inplace=True)
    df_pop_result = df_pop[['구분', '총 인구수']].copy()
    df_pop_result['총 인구수'] = df_pop_result['총 인구수'].astype(int)
    return df_pop_result

def dfCctv(df_cctv):
    df_cctv.drop(index=0, inplace=True)
    df_cctv.rename(columns={'총계':'CCTV 갯수'}, inplace=True)
    df_cctv['구분'] = df_cctv['구분'].apply(delSpace)
    df_cctv['CCTV 갯수'] = df_cctv['CCTV 갯수'].apply(delComma)
    df_cctv_result = df_cctv[['구분', 'CCTV 갯수']].copy()
    df_cctv_result['CCTV 갯수'] = df_cctv_result['CCTV 갯수'].astype(int)
    return df_cctv_result

def dfMerge(x_df, y_df):
    x_df2 = dfPop(x_df)
    y_df2 = dfCctv(y_df)
    return pd.merge(x_df2, y_df2, on ='구분')


def dfPlotScatt(new_df, font_path):
    new_df.rename(columns={'CCTV 갯수':'cctv_갯수', '총 인구수':'총_인구수'}, inplace=True)
    x = new_df.총_인구수.values.reshape(-1,1)
    y = new_df.cctv_갯수
    model = LinearRegression()
    model.fit(x,y)
    new_df['predict_lr'] = model.predict(x)
    
    new_df['res'] = new_df.cctv_갯수 - new_df.predict_lr
    new_df_sort = new_df.sort_values(by='res', ascending=False)
    
    fontprop = font_manager.FontProperties(fname=font_path, size=20)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    scatter = ax.scatter(new_df_sort.총_인구수, new_df_sort.cctv_갯수, c = new_df_sort.res, s=50, label = 'cctv')
    ax.plot(new_df_sort.총_인구수, new_df_sort.predict_lr, color = 'r', label = 'predict')
    
    for n in new_df_sort.index:
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, 'cctv_갯수'], new_df_sort.loc[n, '구분'], fontproperties=fontprop, size =10 )
    plt.colorbar(scatter, ax=ax)
    ax.legend()
    ax.set_xlabel('총 인구수', fontproperties=fontprop)
    ax.set_ylabel('cctv 갯수', fontproperties=fontprop)
    plt.show()