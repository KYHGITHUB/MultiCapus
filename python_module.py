# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%230823
def creatArray(n):
    if 1 <= n <= 30:
        if n == 1:
            a = [[1]]
        else:
            a = [[0 for i in range(n)] for j in range(n)]
            x, y = 0, 0
            vec = 'right'
            for num in range(1, n**2+1):
                a[x][y] = num
                if vec == 'right':
                    if y < n-1 and a[x][y+1] == 0:
                        y += 1
                    elif a[x+1][y] == 0:
                        vec = 'down'
                        x += 1
                elif vec == 'down':
                    if x < n-1 and a[x+1][y] == 0:
                        x += 1
                    elif a[x][y-1] == 0:
                        vec = 'left'
                        y -= 1
                elif vec == 'left':
                    if y > 0 and a[x][y-1] == 0:
                        y -= 1
                    elif a[x-1][y] == 0:
                        vec = 'up'
                        x -= 1
                elif vec == 'up':
                    if x > 0 and a[x-1][y] == 0:
                        x -= 1
                    elif a[x][y+1] == 0:
                        vec = 'right'
                        y += 1
        return a
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.dot(x,w) + b
    if tmp < 0 :
        return 0
    else:
        return 1
    
def step_func(x):
    if x>0:
        return 1
    else:
        return 0

def aryAsbool(x):
    if isinstance(x, np.ndarray):
        y = x>0
        y = y.astype(int)
        return y
    elif isinstance(x, list):
        x = np.array(x)
        return aryAsbool(x)
    else:
        print('list 혹은 array를 입력해주세요')
        
def Plotbool(x):
    plt.plot(x, aryAsbool(x))
    return plt.show()
def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1/(1+np.exp(-x))
    elif isinstance(x, list):
        x = np.array(x)
        return sigmoid(x)

def Plotsigmoid(x):
    if isinstance(x, np.ndarray):
        plt.plot(x, sigmoid(x))
        return plt.show()
    else:
        print('array를 삽입해주세요!')
        
def Plotsigbool(x):
    if isinstance(x, np.ndarray):
        y1 = sigmoid(x)
        y2 = aryAsbool(x)
        plt.plot(x, y1, label='sigmoid')
        plt.plot(x, y2, linestyle='--', label='bool')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-0.1, 1.1)
        return plt.show()
    else:
        print('array를 삽입해주세요')
        
def relu(x):
    return np.maximum(0, x)

def Plotrelu(x):
    if isinstance(x, np.ndarray):
        plt.plot(x, relu(x))
        return plt.show()
    else:
        print('array를 삽입해주세요')

def straightPlot(x):
    if isinstance(x, np.ndarray):
        y = -x + 0.5
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        plt.axvline(x=0,color='k')
        plt.axhline(y=0,color='k')
        plt.scatter([0],[0],marker='o',color='r')
        plt.scatter([1,0,1],[0,1,1],marker='^',color='r')
        plt.xlabel("x") # x축 이름
        plt.ylabel("y") # y축 이름
        plt.fill_between(x,y,-2, alpha=0.5) # y값에서 y축의 -3 까지의 값들에 대해 색깔칠해준다.
        plt.grid()
        return plt.show()
    elif isinstance(x, list):
        y = x[:]
        for idx, i in enumerate(y):
            y[idx] = -i + 0.5
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        plt.axvline(x=0,color='k')
        plt.axhline(y=0,color='k')
        plt.scatter([0],[0],marker='o',color='r')
        plt.scatter([1,0,1],[0,1,1],marker='^',color='r')
        plt.xlabel("x") # x축 이름
        plt.ylabel("y") # y축 이름
        plt.fill_between(x,y,-2, alpha=0.5) # y값에서 y축의 -3 까지의 값들에 대해 색깔칠해준다.
        plt.grid()
        return plt.show()
    else:
        print('array 혹은 list를 삽입해주세요')
def getDat(path):
    df = pd.read_table(path, header=None, sep='::')
    return df

def getDummies(df):        
    col = []
    gen_list = []
    new_df = df.loc[:, :'title']
    
    for genres in df['genres']:
        gen_list.extend(genres.split('|'))
    gen_list = list(set(gen_list))
    gen_list.sort()
    
    for idx1 in range(len(df.index)):
        col.append(pd.get_dummies(df.loc[idx1,'genres']).columns[0].split('|'))
    
    for idx2, j in enumerate(col):
        for gen in gen_list:
            if gen in j:
                new_df.loc[idx2,f'Genres_{gen}'] = 1
            else:
                new_df.loc[idx2,f'Genres_{gen}'] = 0
    return new_df