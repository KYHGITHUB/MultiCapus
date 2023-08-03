# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""

import python_module as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#   %matplotlib -> 로컬에선 쓰지않고 코랩에선 라인에 그래프가 포함되게 해달라는
#                  명령어.
import seaborn as sns
from matplotlib import font_manager, rc
import platform

#%%230803

'''

df = pd.DataFrame(np.arange(6).reshape(3,2),
                  index = range(0,3), columns = ('A', 'B'))
print(df)
print(df.rename(index = {0 : 'a'}))
df.rename(index = {0 : 'T'}, inplace = True)
print(df)
for i in range(3):
    print(f'{df.index[i]}의 타입은 {type(df.index[i])} 이다')
df.rename(columns = {'A':'열번호 1', 'B':'열번호 2'}, inplace =True)
print(df)

font_path = '\\.spyder-py3\\class file\\NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname = font_path, size =10)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000]) #   x축 눈금
labels = ax.set_xticklabels(['하나', '둘', '셋', '넷', '다섯'],        # x축 값들 이름 정하기
                             rotation = 30)     # rotaion : 글자의 기울기
plt.xticks(fontproperties = fontprop)       # fontproperties
plt.show()

property_candidate = {'이재명':3217161, '윤석열':7745343, '심상정':1406297,
                      '안철수':197985542, '오준호':264067, '허경영':26401367, '이백윤':171800,
                      '옥은호':337062, '김동연':4053544, '김경재':2202623, '조원진':2058661,
                      '김재연':51807, '이경희':149907313, '김민찬':421648}
x = list(property_candidate.keys())
y = np.array(list(map(lambda x:x/1000, property_candidate.values())))
print(x)
print(y)
plt.figure(figsize = (8,8))
sns.barplot(x=x, y=y)
sns.set_theme(style='white', context='talk')
plt.xticks(fontproperties = fontprop, rotation = 90)
plt.show()

sns.set_theme(style="white", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(10,8))
fig.subplots_adjust(hspace=0.2)  # adjust space between axes

# plot the same data on both axes
ax1.bar(property_candidate.keys(), list(map(lambda x : x/1000, property_candidate.values())), color = ['yellow', 'cyan', 'pink', 'purple'], alpha = 0.5,
edgecolor = 'black', linewidth = 2.5)
ax2.bar(property_candidate.keys(), list(map(lambda x : x/1000, property_candidate.values())),color = ['yellow', 'cyan', 'pink', 'purple'], alpha = 0.5,
edgecolor = 'black', linewidth = 2.5)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(120000, 200000)  # outliers only
ax2.set_ylim(0, 30000)  # most of the data



# hide the spines between ax and ax2

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax1.set_title('20 대 대선 후보자 재산 현황 (단위 : 백만원)', fontproperties = fontprop, pad=20)
#ax1.set_ylabel('단위 : 백만원',labelpad=20, fontproperties = fontprop)
ax2.xaxis.tick_bottom()
ax2.set_xticklabels(property_candidate.keys(),fontproperties = fontprop,rotation = 270)

#Y축 양쪽에 빗금 넣기
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.show()

plt.rcParams['axes.unicode_minus'] = False 

if platform.system() == 'Darwin':           # paltform -> 윈도우같은걸 말함
    rc('font', family='AppleGothic')        # 초기화 값 읽어주는걸 담당
elif platform.system() == 'Windows':
    path = 'c:\\Windows\\Fonts\\malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    
    font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'       # 로컬에서는 경로 변경이 필요함
    fontprop = font_manager.FontProperties(fname=font_path, size=10)
    font_name = fontprop.get_name()         # get_name : 이름 바꾸기
    rc('font', family=font_name)
print(font_path)

df = pd.Series([3, -8, 2, 0], index=['d', 'b', 'a', 'c'])
print(df)
print(df.reindex(['a', 'b', 'c', 'd', 'e']))
df.reindex(range(4))
df1 = pd.Series(['blue', 'red', 'green'], index =[0, 2, 4])
print(df1)
df1.reindex(range(6))
print(df1.reindex(range(6), method = 'ffill')) #   ffill : f + fill 로서 앞에껄로 채운다는 의미
df = pd.DataFrame(np.arange(6).reshape(3,2),index = range(0, 5, 2), columns = ('A', 'B'))
df_ext = df.reindex(range(5), columns = ['B', 'C', 'A'])
print(df_ext)
df_ext.drop(1, inplace = True)  # indeex 지우기
print(df_ext)
print(df_ext.drop(0))
print(df_ext.drop('C', axis = 'columns'))  # axis : 축, axis = 'columns' : 축이 columns축이다 
print(df_ext.drop('C', axis = 1))
df_ext.drop(3, inplace=True)
print(df_ext)
df = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(df)
print(df[1])
print(df['b'])
print(df[:3])
print(df['a':'c'])     # 문자로 슬라이싱 할때는 숫자와는 달리 문자 -1 이 아닌 문자까지 간다. ex) x[5] -> 인덱스 4번까지, x['e'] -> 인덱스값이 'e'인곳 까지
df['c':'d'] = 0
print(df)
df = pd.DataFrame(np.arange(3*4).reshape(3, 4), index = ['A', 'B', 'C'], columns = ['aa', 'bb', 'cc', 'dd'])
print(df)
print(df['aa'])
print(df[['aa', 'cc', 'dd']])
print(df[:2])
print(df[df['aa']<=4])
print(df.loc[['A','B'], ['aa', 'cc']])     #   loc[행, 열]
print(df)
print(df.loc['A'])
print(df.iloc[:2, [0, 2]])

url = 'https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv'
df = pd.read_csv(url, sep='\t')
print(df)
df.info()
print(df.shape)
print(list(df.columns))
for i in df.columns:
    print(i)
'''
path = 'D:\\.spyder-py3\\class file\\example_data\\example_data\\'
df = pd.read_csv(path + 'ex1.csv', sep = ',')
print(df)
