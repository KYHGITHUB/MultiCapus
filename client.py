import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
import pandas as pd
pd.set_option('display.max_columns', None)      # 데이터프레임 끝까지 보여주기
pd.set_option('display.max_rows', None)


#%%230808
df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']], index = ['준서', '예은'], columns = ['나이', '성별', '학교'])
df
print(df.index)
print(df.columns)
df.index = ['학생1', '학생2']
df.columns = ['연령', '남녀', '소속']
print(df)
df.rename(columns = {'남녀':'sex'}, inplace = True)
print(df)
df2 = df.copy()
df2.drop(index = '학생1', inplace = True)
print(df2)
print(df)
df.loc['학생1', 'sex'] = '여'
print(df)
df['소속']
exam_data = {'이름':['서준', '우현', '인아'],
             '수학':[90, 80, 70],
             '영어':[98, 89, 95],
             '음악':[85, 95, 100],
             '체육':[100, 90, 90]}
df = pd.DataFrame(exam_data)
df
df.sort_values(by = '음악')
df
df2 = df.sort_values(by = '체육')
df2
df2.index = ['a', 'b', 'c']
df2
df2.loc['a']
df2
print(df2.reset_index())
df2
print(df2.loc['a', ['음악','체육']])
print(df2.loc['a', '음악':'체육'])
df2.set_index('이름', inplace = True)
df2
print(df2.loc['인아':'서준', '수학':'음악'])
df
df.set_index('이름', inplace = True)
df
df3 = df.loc[:, '수학':'영어']
df3.reset_index(inplace = True)
print(df3)
