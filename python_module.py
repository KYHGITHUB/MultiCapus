# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import copy
from decimal import Decimal




#%%230726
def f1(x):
    return x*x + 3*x - 10
def f2(x):
    return x*x*x
def g(func):
    return [func(x) for x in range(-2, 3)]
def increment(n):
    return n+1
def square(n):
    return n**2
def f(x):
    return x*x
def change_values(ch_list, before, after):
    copy_ch_list = copy.deepcopy(ch_list)
    for i in range(len(copy_ch_list)):
        if type(copy_ch_list[i]) != list:
            if copy_ch_list[i] == before:
                copy_ch_list[i] = after
        else:
            copy_ch_list[i] = change_values(copy_ch_list[i], before, after)
            #change_values(copy_ch_list[i], before, after)
    return copy_ch_list
def change_values2(L, from_s, to_s):            #작동 제대로 안함
    for k, ele in enumerate(L):
        if ele == from_s:
            L[k] = to_s
        elif type(ele) == list:
            change_values2(ele, from_s, to_s)
    return L
def change_values3(ch_list, before, after):         #작동 제대로 안함
    for i in range(len(ch_list)):
        if type(ch_list[i]) != list:
            if ch_list[i] == before:
                ch_list[i] = after
        else:
            change_values3(ch_list[i], before, after)
    return ch_list

def frange(start, stop=0, step=1.0):
    frange_list = []
    if stop == 0:          
        stop = start     # frange(9) -> frange(9, 0) -> frange(9, 9) -> frange(0, 9)
        start = 0.0
    val = float(start)
    if start < stop:
        while val < stop:
            frange_list.append(val)
            val += step
    elif start > stop:
        while val > stop:
            frange_list.append(val)
            val += step
    return frange_list

def frange2(start, *args):
    if len(args) == 0:
        stop = start
        start = Decimal('0.0')
        step = Decimal('1.0')
    elif len(args) == 1:
        stop = Decimal(str(args[0]))
        step = Decimal('1.0')
    elif len(args) == 2:
        stop = Decimal(str(args[0]))
        step = Decimal(str(args[1]))

    frange_list = []
    if start > stop:
        while start > stop:
            frange_list.append(float(start))
            start += step
    elif start < stop:
        while start < stop:
            frange_list.append(float(start))
            start += step
    return frange_list

def frange3(arg1, *args):
    if len(args) == 0:
        start, stop, step = 0.0, float(arg1), 1.0
    if len(args) == 1:
        start, stop, step = arg1, float(args[0]), 1.0
    if len(args) == 2:
        start, stop, step = arg1, float(args[0]), float(args[1])
    
    frange_list = []
    if start > stop:
        while start > stop:
            frange_list.append(start)
            start += step
    if start < stop:
        while start < stop:
            frange_list.append(start)
            start += step
    return frange_list

def str_test():
    import string
    print(string.punctuation)