# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:06:13 2023

@author: rnjsd
"""

import requests
from datetime import datetime as dt
import json
import pandas as pd


def get_movie():
    url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=c3a0fa760324363ef7cde2afa0d73297&targetDt='
    resp = requests.get(url+'20230807')
    data = resp.json()['boxOfficeResult']['dailyBoxOfficeList']
    dataframe_of_movie = pd.DataFrame(data)
    df = dataframe_of_movie.loc[:, ['rank', 'movieNm', 'openDt','salesAmt']]
    return df
