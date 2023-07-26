# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""
import python_module as pm
import mymath
#from mymath import area, mypi
#from mymath import *
from string import punctuation as punc

#%%230726
f = lambda x, y : x+y
print(f(1, 2))
print(f('test', 'python'))
print(f([1,2], [3,4]))
vargs = lambda x, *args : args
print(vargs(2,3,4,5,6,7))
print(pm.g(pm.f1))
print(pm.g(pm.f2))
print(pm.g(lambda x : x*x + 3*x - 10))
func_list = [
    lambda x, y : x + y,
    lambda x, y : x - y,
    lambda x, y : x * y,
    lambda x, y : x / y
]
for i in func_list:
    print(i(2,1))
    
plus = lambda x, y : x + y
minus = lambda x, y : x - y
multiply = lambda x, y : x * y
divide = lambda x, y : x / y

list_operation = [plus, minus, multiply, divide]
for i in list_operation:
    print(i(6, 3))    

for data, action in [(2, pm.increment), (4, pm.square)]:
    print(action(data))

data = [1, 2, 3, 4]
print([pm.f(i) for i in data])

list_result = []
for i in data:
    list_result.append(pm.f(i))
print(list_result)

result_map = map(pm.f, data)
type(result_map)
print(result_map)
print(list(result_map))
print(list(map(pm.f, data)))
print(list(map(lambda x : x*x, data)))
print(list(map(lambda x : x*x + 3*x +5, range(10))))

data_list = ['hello', 'python', 'programming']
print(list(map(lambda x : len(x),data_list)))

xx = list(range(1, 6))
yy = list(range(6,11))
z = list(map(lambda x, y : x+y, xx, yy))
z_map = map(lambda x, y : x+y, xx, yy)
print(z)
print(next(z_map))
print(next(z_map))
print(next(z_map))
print(next(z_map))
print(next(z_map))

print(list(filter(lambda x : x>3, [2,3,4,5,6])))
print(type(filter(lambda x : x>3, [2,3,4,5,6])))

list_3 = []
for i in [2,3,4,5,6]:
    if i>3:
        list_3.append(i)
print(list_3)

print(list(filter(lambda x : x%2==0, range(1,11))))
print(list(filter(lambda x : x%2!=0, range(1,11))))

list(filter(lambda x : x%2-1, range(1,11)))

list_ = ['high', 'level', '', None, 'builtint', 'func']
print(list(filter(None, list_)))       # None = 아무런 조건식이 없어, 입력값 자체를 진릿값으로 사용한다는 의미로 해석)

fnames = ['a_thumb.jpg', 'b01_thumb.jpg', 's100_thumb.jpg', 's100.jpg', 'b01.jpg']
print(list(filter(lambda x : 'thumb' in x, fnames)))
print(list(filter(lambda x : 'thumb' not in x, fnames)))

L = [3, 2, [3, [[3], 4]]]
print(pm.change_values(L, 3, 5))
print(pm.change_values2(L, 3, 5))
print(pm.change_values3(L, 3, 5))
L = [3, 2, [3, [[3], 4]]]
print(pm.frange(4))
print(pm.frange(2, 7))
print(pm.frange(1, 3, 0.2))
print(pm.frange(3, 1, -0.2))
print(pm.frange2(4))
print(pm.frange2(2, 7))
print(pm.frange2(1, 3, 0.2))
print(pm.frange2(3, 1, -0.2))
print(pm.frange3(4))
print(pm.frange3(2, 7))
print(pm.frange3(1, 3, 0.2))
print(pm.frange3(3, 1, -0.2))
print(punc)
print(pm.str_test())
