import python_module as pm
import numpy as np


#%%230823
x = np.arange(-5, 5, 0.1)
y =list(x)
pm.Plotbool(x)
pm.Plotsigmoid(x)
y = list(range(10))
pm.Plotbool(y)
pm.Plotsigmoid(y)


pm.Plotsigbool(x)

np.linspace(0, 3, 1000)

pm.straightPlot(x)
pm.straightPlot(y)
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = list(range(10))
pm.straightPlot(x)
pm.straightPlot(y)

x

type(np.array(y))
pm.aryAsbool(y)


file_path = 'D:\\.spyder-py3\\class file\\'
file_name = 'movies.dat'
df = pm.getDat(file_path + file_name)
name = ['movie_id', 'title', 'genres']
df.columns = name
print(df)
new_df = pm.getDummies(df)
print(new_df)
