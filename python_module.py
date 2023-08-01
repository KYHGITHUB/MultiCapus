# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""

class Song:
    def set_song(self, title, category):
        self.t = title
        self.c = category
    def print_song(self):
        print(f"노래제목 : {self.t}({self.c})")

class Singer(Song):
    def set_singer(self, name):
        self.name = name
    def hit_song(self, song):
        self.hits = song
    def print_singer(self):
        print(f'가수이름 : {self.name}')
        self.hits.print_song()

class Personal(list):
    def __init__(self, name, dob, descs):
        self.name = name
        self.dob = dob
        self.append(descs)
    def present(self):
        print(self.name, ":", self.dob)
        print(self)

def wrapper1(func):
    def wrapped_func():
        print('====before====')
        func()
        print('======after=====')
    return wrapped_func
def myfunc1():
    print('   I am here')
@wrapper1
def myfunc2():
    print('   me too')

def makebold(fn):
    def wrapped():
        return '<b>' + fn() + '</b>'
    return wrapped
def makeitalic(fn):
    def wrapped():
        return '<i>' + fn() + '</i>'
    return wrapped
@makeitalic
@makebold
def say():
    return 'Hello'

def debug1(fn):
    def wrapped(a, b): # fn과 동일한 인수를 적어준다
        print('debug', a, b)
        return fn(a, b) # 함수 fn을 호출한다. 
    return wrapped
@debug1
def add1(a, b):
    return a + b

def debug2(fn):
    def wrapped(*args, **kwargs):
        print('calling', fn.__name__, 'args= ', args, 'kwargs= ', kwargs)
        result = fn(*args, **kwargs)
        print('    result = ', result)
        return result
    return wrapped
@debug2
def add2(a, b):
    return a +b

def wrapper3(fn):
    def wrapped(*args):
        print(f'====befor====')
        result = fn(*args)
        print(f'====after====')
        return result
    return wrapped
@wrapper3
def myfunc3(a, b):
    print('I am here.')
    return a + b
@wrapper3
def myfunc4(a, b):
    print('안녕')
    return a*b
