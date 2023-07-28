# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
from time import time, ctime, sleep


class MyClass:
    def __add__(self, x):
        print(f'add {x} called')
        return x
class MyClass2:
    def __init__(self, name, age):
        self.name = name
        self.age = age
class MyClass3:
    def __init__(self):
        self.age = 21
        self.name = '홍길동'
class MyClass4:
    def __init__(self, age=21, name='홍길동'):
        self.age = age
        self.name = name
class MyClass5:
    def __init__(self, phone, age=21, name='홍길동'):
        self.phone = phone
        self.age = age
        self.name = name

    def print_attr(self, number):
        self.number = number
        print(self.phone)
        self.phone = self.number
        return self.phone
class A:
    def f(self):
        print('base')
class B(A):
    pass
class MyClass6:
    def set(self, v):
        self.value = v
    def get(self):
        return self.value
    def tem(self):
        self.value2 = self.get()
        print(self.value2)
class Life:
    def __init__(self):
        self.birth = ctime()
        print('Birthday', self.birth)
    def __del__(self):
        print('Deathday', ctime())
class Var:
    c_mem = 100 # 클래스 멤버
    def f(self):
        self.i_mem = 200 # 인스턴스 멤버
    def g(self):
        return self.i_mem, self.c_mem
class FourCal:
    def __init__(self, first, second):
        self.setdata(first, second)
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        return self.first + self.second
    def mul(self):
        return self.first * self.second
    def sub(self):
        return self.first - self.second
    def div(self):
        return self.first / self.second
class safeFourCal(FourCal):
    def div(self):
        if self.second == 0:
            return 0
        return self.first / self.second