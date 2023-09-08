import python_module as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from bs4 import BeautifulSoup as bs
import requests
from scipy import stats

#%%230908

file_path = 'D:\\.spyder-py3\\class file\\'
df = pd.read_csv(file_path+'survey.csv')
print(df.income.mean())
print(df.income.median())

male = df['income'][df['sex'] == 'm']
female = df['income'][df['sex'] == 'f']


# t 검정

print(stats.ttest_ind(male, female))	# t 검정 - H0 : 두 모집단의 평균 간에 차이가 없다.

# 등분산 검정 만족하는 t 검정
result = stats.ttest_ind(male, female) # 등분산 검정 만족하는 경우
alpha = 0.05
p_value = result[1]
print(p_value)
if p_value < alpha:
    print('귀무가설을 기각한다. 두 평균에 차이가 있습니다')
else:
    print('귀무가설을 채택한다. 두 평균에 차이가 없습니다')

# 등분산 검정 만족하지 못하는 t 검정
result2 = stats.ttest_ind(male, female, equal_var=False)  # 이분산의 방식 => 등분산 검정 만족하지 못함.
p_value = result2[1]
if p_value < alpha:
    print('두 평균에 차이가 있습니다')
else:
    print('두 평균에 차이가 없습니다')

# 등분산 검정
stat, p_value = stats.levene(male, female)	# Levelne 등분산 검정 - H0 : 두 집단의 분산은 같다
print(stat)
print(p_value)
alpha = 0.01
if p_value < alpha:
    print('귀무가설을 기각한다. 두 분산이 동일하지 않음. p-value :', p_value)
else:
    print('귀무가설을 채택한다. 두 분산이 동일함. p-value :', p_value)

# 정규성 검정
result = stats.shapiro(male) # 샤피로-윌크 검정 - H0 : 정규 분포를 따른다
alpha = 0.05
p_value = result[1]
if p_value < alpha:
    print('귀무가설을 기각한다. male 데이터는 정규 분포를 따르지 않는다.')
else:
    print('귀무가설을 채택한다. male 데이터는 정규 분포를 따른다.')

result = stats.shapiro(female) # 샤피로-윌크 검정 - H0 : 정규 분포를 따른다
alpha = 0.05
p_value = result[1]
if p_value < alpha:
    print('귀무가설을 기각한다. female 데이터는 정규 분포를 따르지 않는다.')
else:
    print('귀무가설을 채택한다. female 데이터는 정규 분포를 따른다.')

# 회귀분석
rv = stats.norm(2, 0.5) # norm:정규분포, loc:평균, scale:표준편차
x = np.arange(0, 4.1, 0.1)
y = rv.pdf(x)
plt.plot(x, y, lw=5) # lw : 그래프 선의 굵기
plt.grid()
plt.show()
print(rv.cdf(1.7)) # 그래프의 왼쪽에서부터 x=1.7 까지의 넓이 (=확률)
print(rv.isf(1-0.27425311775007355)) # 그래프의 오른쪽을 기준으로 넓이 a를 만족하는 x축값. -> 전체넓이는 1

# 넓이 0.9인 그래프의 가운데 구간을 구해보자

print(f'시작 x값 : {rv.isf(0.95)}')
print(f'끝 x값 : {rv.isf(0.05)}')