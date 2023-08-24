import python_module as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#%%230824
sr = pd.Series(range(5))
pm.catCodes(sr)
if isinstance(df_di, pd.DataFrame):
    print('시리즈입니다')
iris = sns.load_dataset('iris')
iris
fig = plt.figure(figsize=(6, 4))
df
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
print(pm.catCodes(fruits))
di = {'alphabet':['a','b','c','d','a','b',
                 'b','b','e','t'],
      'number':[1,2,3,1,1,1,5,7,3,3]}
pm.catCodes(di)
df_di = pd.DataFrame(di)
type(df_di)
df_di['alphabet'].astype('category').cat.codes
x=[{'city':'seoul', 'temp':10.0}, {'city':'Dubai', 'temp':33.5}, {'city':'LA', 'temp':20.0}]
x, x_rare= pm.dictAsvec(x)
print(x_rare)
D = [{'foo':1, 'bar':2}, {'foo':5, 'baz':4}]
pm.dictAsvec(D)
d = pd.DataFrame()
text=['떴다 떴다 비행기 날아라 날아라',
      '높이 높이 날아라 우리 비행기',
      '내가 만든 비행기 날아라 날아라',
      '멀리 멀리 날아라 우리 비행기']
if isinstance(df_di, pd.DataFrame):
    print('시리즈입니다')
pm.textAsvec(text)
text1 = ['''떴다 떴다 비행기 날아라 날아라
          높이 높이 날아라 우리 비행기
         내가 만든 비행기 날아라 날아라
        멀리 멀리 날아라 우리 비행기''']
d['al'] = [1,2,3,4]
d['sr'] = sr
d
text1
pm.textAsvec(text1)
df = pm.textAsimport(text1)
text2 = '''떴다 떴다 비행기 날아라 날아라
        높이 높이 날아라 우리 비행기
        내가 만든 비행기 날아라 날아라
        멀리 멀리 날아라 우리 비행기'''
text2
df = pm.textAsimport(text2)

df_di = pd.DataFrame(di)
df_di
pm.catCodes(df_di)
df_di

