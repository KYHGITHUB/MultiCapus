# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as dv
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tv

#%%230824
def catCodes(data):
    if isinstance(data, (list, pd.Series)):
        df = pd.DataFrame(data)
        cat_df = df[0].astype('category')
        return cat_df.cat.codes, cat_df.cat.categories
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
        new_df = df.copy()
        category_info = {}
        for column in df.columns:
            cat_df = df[column].astype('category')
            new_df[column] = cat_df.cat.codes
            category_info[column] = cat_df.cat.categories
        return new_df, category_info
    elif isinstance(data, pd.DataFrame):
        category_info = {}
        new_df = data.copy()
        for column in data.columns:
            cat_df = data[column].astype('category')
            new_df[column] = cat_df.cat.codes
            category_info[column] = cat_df.cat.categories
        return new_df, category_info
def dictAsvec(dic, bools=True): 
    if bools:
        vec = dv()
        vec_rare = vec.fit_transform(dic)
        vec_ary = vec_rare.toarray()
        vec_names = vec.get_feature_names_out()
        df = pd.DataFrame(vec_ary, columns=vec_names)
        return df, vec_rare
    else:
        vec = dv(sparse=bools)
        vec_ary = vec.fit_transform(dic)
        vec_names = vec.get_feature_names_out()
        df = pd.DataFrame(vec_ary, columns=vec_names)
        return df
    
    
def textAsvec(text):
    if isinstance(text, list):
        vec = cv()
        txt_rare = vec.fit_transform(text)
        txt_ary = txt_rare.toarray()
        txt_names = vec.get_feature_names_out()
        df = pd.DataFrame(txt_ary, columns=txt_names)
        return df, txt_rare
    else:
        new_text = (list(text))
        return textAsvec(new_text)
def textAsimport(text):
    if isinstance(text, list):
        vec=tv()
        txt_rare = vec.fit_transform(text)
        txt_ary = txt_rare.toarray()
        txt_names = vec.get_feature_names_out()
        df = pd.DataFrame(txt_ary, columns=txt_names)
        return df, txt_rare
    else:
        lines = text.split('\n')
        new_text = [line.strip() for line in lines]
        return textAsimport(new_text)