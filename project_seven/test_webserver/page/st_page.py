# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:51:13 2023

@author: rnjsd
"""
from utils import st_mo as stm
import streamlit as st

def app():
    st.write('1번을 선택했습니다.')
    df = stm.get_movie()
    st.dataframe(df)
