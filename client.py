import python_module as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from bs4 import BeautifulSoup as bs
import requests

#%%230905

url = 'https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'
resp = requests.get(url)
soup = bs(resp.text, 'lxml')
rows = soup.select('div > ul > li')
etfs = pm.readweb(rows)
df = pd.DataFrame(etfs)

print(etfs.keys())
print(df)