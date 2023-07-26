# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 17:09:12 2023

@author: rnjsd
"""
#%%230726
mypi = 3.14

def add(a, b):
    return a+b

def area(r):
    return mypi*r*r

if   __name__ == "__main__" :      # __name__ : 변수, "__main__" : 값
    print(area(4.0))