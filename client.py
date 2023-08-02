# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""

import python_module as pm
from  datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#   %matplotlib -> 로컬에선 쓰지않고 코랩에선 라인에 그래프가 포함되게 해달라는
#                  명령어.
import copy



'''
#%%230802
#%% datetime
dt = datetime(2023, 8, 2, 13, 5, 20)
print(dt)
print(dt.year)
print(dt.month)
print(dt.date())
print(dt.time())
print(dt.date(), dt.time())

print(dt.strftime('%Y%m%d %H:%M'))
datetime.strptime('20230802', '%Y%m%d')
dt
print(dt.replace(minute=5, second=47))
dt = datetime.now()
dt.microsecond
delta = datetime.now() - dt
print(delta)
print(type(delta))
print(dt + timedelta(hours=2))
print(dt + timedelta(30))

print(dir(timedelta))
print(f'현재시간 - 과거시간 = {delta.seconds}')
base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
print(f'base_time : {base_time}')
future_time = datetime(2023, 12, 26)
print(f'future_time : {future_time}')
print(f'(future_time - base_time).total_seconds() = {(future_time - base_time).total_seconds()}')
diff = future_time - base_time
print(diff)
print(diff.total_seconds() / 3600)
for i in range(diff.days):
    print(i)

#%%     
x_seq = list(range(10))
data = np.arange(10)
print(x_seq)
print(data)
for i in x_seq:
    print(i*3)
y = data ** 2
x = data
plt.plot(data, data**2)
plt.show()

fig = plt.figure()
axes = fig.add_subplot()
axes.plot(x, y)
plt.show()

fig, axes = plt.subplots()
axes.plot(x, y)
plt.show()

x = np.arange(-10, 11, 1)
x
fig, ax = plt.subplots(figsize = (4, 4))
plt.plot(x, x*x)
plt.plot(x, 2*x)

t = np.arange(0, 5, 0.5)
plt.figure(figsize=(10,6))
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'gs')
plt.plot(t, t**3, 'b>')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(np.random.randn(50).cumsum(), 'k--')       # randn : 정규분포에서 추출된 난수들, cumsum : 누적 합계
_ = ax1.hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
ax2.scatter(np.arange(30), np.arange(30)+ 3 * np.random.randn(30))

fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace=0, hspace=0)

x = np.arange(10)
y = x*10 + 2
fig, ax = plt.subplots()
ax.plot(x, y, 'g--')

fig, ax = plt.subplots()
ax.plot(x, y, linestyle = '--', color = 'g')
plt.plot(np.random.randn(30).cumsum(), 'ko--')
plt.plot(np.random.randn(30).cumsum(), color = 'k', linestyle = 'dashed', marker = 'o')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30)
ax.set_title('My first plot', fontsize = 18)
ax.set_xlabel('Stages', fontsize = 13)

props = {'title' : 'My first plot', 'xlabel' : 'Stages'}
ax.set(**props)
#%%
x = np.arange(1, 10, 0.1)
loss = lambda x : np.exp(-x)
acc = lambda x : -np.exp(-x)

x1 = np.random.randn(len(x))

fig, loss_ax = plt.subplots(figsize=(8,6))

acc_ax = loss_ax.twinx() # x 축을 공유하는 새로운 axes 객체를 만들어 준다. 결과적으로 x축은 같고 y측만 다른 그래프가 생긴다.

loss_ax.plot(x, loss(x), 'y', label = 'train loss')
loss_ax.plot(x, loss(x-x1/5), 'r', label='validation loss')

acc_ax.plot(x, acc(x), 'b', label='tarina acc')
acc_ax.plot(x, acc(x-x1/7), 'g', label='val acc')
for label in acc_ax.get_yticklabels(): # y 축 tick 색깔 지정
    label.set_color("blue")

loss_ax.set_xlabel('epoch', size=20)
loss_ax.set_ylabel('loss', size=20)
acc_ax.set_ylabel('accuray', color='blue', size=20)

loss_ax.legend(loc='upper right')
acc_ax.legend(loc='lower right')
fig.savefig('test1.png')
plt.show()
#%%
x = np.arange(0,10, np.pi/100)
f = lambda x : np.sin(x)+x/10

XTN=[r'$0 \pi$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$',r'$5\pi/2$',r'$3\pi$']
plt.figure(figsize=(8,6))
plt.plot(x, f(x))
plt.title('Plot Exercise',position=(0.5,1.04), fontsize=20)
plt.xlabel (r'$x$',fontsize=20)
plt.ylabel (r'$f (x) = sin(x) + x$', fontsize=20)
plt.xticks(np.arange(0.0,10.0,np.pi/2), labels=XTN, fontsize=15)
plt.yticks(np.arange(-1,2.1,np.pi/5), fontsize=15)
plt.text(0.8,-0.3,'Test Messages', color='k', fontsize=18)
plt.grid()  # 배경에 줄 쳐주는것
plt.tight_layout()  # title이 있으면 그래프 외곽선에 맞추기
plt.savefig('test2.png')
plt.show()
'''

x = [10, 20 ,30]
array_x = np.array(x)
print(array_x)
obj = pd.Series(array_x)
print(obj)
print(type(obj))
print(len(dir(obj)))
print(obj.values)
print(type(obj.values))
print(obj.index)
for i in obj.index:
    print(i, obj[i])
obj2 = pd.Series([4, 7, -5, 3], index = ['a', 'b', 'c', 'd'])
print(obj2)
print(obj2.index)
for i in obj2.index:
    print(i, obj2[i])
print(obj2['a'])
obj2['d'] = 6
print(obj2)
print(obj2[obj2>0]) # 외우기
data = pd.Series([-1, 0 ,1], index=['a','b','c'])
print(data[data<0])
print(obj2**2)
print(data*2)
print(np.exp(data))
print('c' in data)
print(1 in data)
print(data[0])
sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':1600, 'Utah':5000}
print(type(sdata))
obj3 = pd.Series(sdata)
print(obj3)
print(obj3['Ohio'])
print(list(obj3.index))
std__ = ['California', 'Ohio', 'Oregon', 'Texas']
sdata
obj4 = pd.Series(sdata, index = std__)
print(obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj4.isnull())
print(obj4.notnull())
obj3
obj4
print(obj3 + obj4) # NaN == None
obj4.name = 'population'
print(obj4)
obj4.index.name = 'state'
obj
obj.index = ['Bob', 'Stetve', 'Jeff']
print(obj)

data = {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year':[2000, 2001, 2002, 2001, 2002, 2003],
        'pop':[1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
        }
df = pd.DataFrame(data)
print(df)
인구1 = {'인구':{'하나':1542, '여섯':9776, '둘':1535},
         '지역':{'하나':'대전', '여섯':'서울', '둘':'대전', '셋':'대전'}}
print(인구1)
print(인구1.keys())
print(인구1['인구'].keys())
print(pd.DataFrame(data, columns=['year', 'state', 'seoul', 'pop']))
df2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'],
                   index = ['one', 'two', 'three', 'four', 'five', 'six'])
print(df2)
print(df2.index)
print(df2.columns)
print(type(df2))
print(type(df2['state']))
print(df2['state'])
print(df2.state)   # == df2['state']
print(df2['pop'])
df5 = pd.DataFrame(인구1)
print(df5)
print(df5.인구)
print(df5['인구'])
print(df5.loc['여섯'])
print(df5.iloc[1])
df2['debt'] = 16.5
print(df2)
copy_df2 = copy.copy(df2)
copy_df2['debt'] = ['틀렸어', 'a', 1, 'spiderman', 'superman', 7]
print(copy_df2)
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
print(val)
df2['debt'] = val
print(df2)
df2['stern'] = df2['state'] == 'Ohio'
print(df2)
del df2['stern']
print(df2)
print(data)
pops = {'Nevada':{2001:2.4, 2002:2.9}, 'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}
df3 = pd.DataFrame(pops)
print(df3)
print(df3.T)
print(pd.DataFrame(pops, index=[2001, 2002, 2003]))
print(df3['Ohio'][:-1])
pdata = {'Ohio':df3['Ohio'][:-1],
         'Nevada':df3['Nevada'][:2]}
print(pdata)
print(pd.DataFrame(pdata))
print(df5)
df5.index.name = '번호'
df5.columns.name = '분류'
print(df5)

data = range(3)
s1 = pd.Series(data, index = ['a', 'b', 'c'])
s2 = pd.Series(data, index = ['a', 'b', 'c'])
s3 = pd.Series(data, index = ['a', 'b', 'c'])
print(s1)
print(s2)
print(s3)
print(s1.rename(index = {'a':'A'}))
s2.index = '구분'.join(s2.index).replace('a', 'A').split('구분')
s3.index.values[0] = 'A'
print(s2)
print(s3)
df5
print('인구' in df5.columns)
print('하나' in df5.index)
print(df5.loc['하나'])
df = pd.Series([3, -8, 2, 0], index=['d', 'b', 'a', 'c'])
print(df)
df.reindex(['a', 'b', 'c', 'd', 'e'])
print(np.arange(6).reshape(3,2))   #   reshape(row, columns)
df = pd.DataFrame(np.arange(6).reshape(3,2), index = range(0, 5, 2), columns = ['A', 'B'])
print(df)
