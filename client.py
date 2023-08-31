import python_module as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web

name = '삼성전자'
code = pm.findCodes(name)
df = web.DataReader(code, 'naver', start='2012-01-01', end='2023-08-29')

pm.prePlot(df, 'Close', '2023-04-30', 20)
pm. prophetPlot(df, 'Close', '2023-04-30', 107)


file_path = 'D:\\.spyder-py3\\class file'
df = pd.read_table(file_path + '\\temperature_ts_data', sep=',')
new_df = pm.timeDfAgg(df, 'timestamp', 'temperature', 'mean', 'size')
print(new_df)









