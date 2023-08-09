# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:24:07 2023

@author: rnjsd
"""
import streamlit as st
from datetime import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from utils import st_mo as stm
from page import st_page as spg
from page import folium_map as fum

today = dt.today()


st.title('시험용 메세지 입니다. title')
new_today = today.strftime('%m월 %d일')
st.header(f"오늘은 {new_today} 입니다")

item = st.sidebar.selectbox('아무나 메뉴입니다[', ['선택1', '콤보2', '세트3'])
if item == '선택1':
    spg.app()
elif item == '콤보2':
    st.write(' 2번을 선택했습니다. ')
    df = pd.DataFrame(np.arange(15))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(df)
    st.pyplot(fig)
elif item == '세트3':
    fum.app()

