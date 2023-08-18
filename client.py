import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
import pandas as pd
pd.set_option('display.max_columns', None)      # 데이터프레임 끝까지 보여주기
pd.set_option('display.max_rows', None)


