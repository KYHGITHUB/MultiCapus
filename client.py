import python_module as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pwd
file_path = 'D:\\.spyder-py3\\class file'
tips = pd.read_csv(file_path + '\\tips.csv', sep=',')
gpd = tips.groupby(['day', 'smoker'])
for i, group in gpd:
    print(i)
group



df_cctv = pd.read_excel(file_path+'\\12_04_08_E_CCTV.xlsx')
df_pop = pd.read_csv(file_path+'\\주민등록인구_20230828092210.csv')



pm.PredictModelPlot(df_cctv, df_pop)















