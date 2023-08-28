import python_module as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_path = 'D:\\.spyder-py3\\class file'
df_cctv = pd.read_csv(file_path + '\\서울시 자치구'
                      ' 년도별 CCTV 설치 현황_230630기준.csv', encoding='euc-kr') 

df_pop = pd.read_csv(file_path + '\\주민등록인구_20230828092210.csv')
new_df = pm.dfMerge(df_pop, df_cctv)
print(new_df)
new_df.info()
font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'
pm.dfPlotScatt(new_df, font_path)
