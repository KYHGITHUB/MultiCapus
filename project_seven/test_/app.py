# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:31:54 2023

@author: rnjsd
"""

import streamlit as st
from datetime import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.title('멀티캠퍼스')
title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)

name = st.text_input('사용자')
st.write(name)
