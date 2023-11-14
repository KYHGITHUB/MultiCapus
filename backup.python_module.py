# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import copy
import random
import hashlib
import os
from decimal import Decimal
import folium
import json
from time import time, ctime, sleep
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as dv
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.linear_model import LinearRegression as LR
from prophet import Prophet
from datetime import datetime
from matplotlib import font_manager
import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score # 모델 평가
from sklearn.ensemble import RandomForestClassifier
#%% 230713
print('out layer')
def test():
    print('ok')

def adds(data_list):
    sums = 0
    for i in data_list:
        sums += i
    return(sums)

def print_list(file_lists):
    for i,j in enumerate(file_lists):
        print(i, j)
        
def adds1(n):
    sums = 0
    for i in range(1, n+1):
        sums += i
    return sums

def create_dict(n):
    return {i: i**2 for i in range(1, n+1)}
    
def create2_dict(n):
    num_dict = {}
    for i in range(1, n+1):
        num_dict[i] = i**2
    return num_dict

def colab_contents_230713():
        a_list = [1, 2, 3, 4]
        a_list[2] = 10
        print(a_list)
        file_list = ['sample_data/README.md',
                     'sample_data/read',
                     'sample_data/output',
                     'sample_data/mnist_test.csv',
                     'sample_data/mnist_train_small.csv',
                     'sample_data/california_housing_test.csv']
        f_tuple = tuple(file_list)
        print(f_tuple)
        print(f_tuple[:3])
        def print_list(file_lists):
            for i,j in enumerate(file_lists):
                print(i, j)
        print_list(f_tuple)
        s_list = [1,2,'python']
        s_list.extend('python')
        s_list.extend(['python'])
        s_list.extend(['python'])
        print(s_list)
        s_list.extend(('python'))
        print(s_list)
        s_list.extend(('python',))  #튜플 합치기, 중요함
        print(s_list)
        f_list = []
        for i in file_list:
            f_list.append(i[12:])
        f_tuple = tuple(f_list)
        print(f_tuple)
        print('read' in f_tuple)
        print(f_tuple[2])
        (*green, yellow, red) = f_tuple  # unpacking
        print(green)
        print(red)
        y = list(f_tuple)
        y[3] = 'yellow'
        f_tuple = tuple(y)
        print(f_tuple)
        def test_func():
            return (32, 34)
        print(test_func())
        x = 1, 2, 3, 4, 5  # packing
        print(x)
        print(type(x))
        print(f_tuple + (1, 2, 3))
        print(f_tuple.count('yellow'))
        print(f_tuple.index('output'))
        thisset = {1,2,3,4,4,4,4,4,5,6,5,5,5,5,5,6,6,6,7,8,9}
        print(type(thisset))
        print(thisset)
        a = {1:100,2:200,3:300,4:400}
        print(a)
        print(list(a.keys())[0])
        print(a.keys())
        print(a[3])
        for i in a:
            print(i, a[i])
        sample_dict = {}
        print(type(sample_dict))
        print(sample_dict.keys())
        sample_dict['one'] = 1
        print(sample_dict)
        sample_dict['two'] = 2
        sample_dict[3] = 'three'
        print(sample_dict)
        sample_dict = dict(three=3, four=4)  # 딕셔너리 만드는방법 중요!
        print(sample_dict)
        for i in sample_dict.items(): # key와 values 묶어서 도출
            print(i)
        sample_dict['color'] = {'red':(255, 0, 0), 'green':(0, 255, 0), 'blue':(0, 0, 255)}
        print(sample_dict)
        print(sample_dict['color']['blue'][2])
        print(sample_dict.keys())
        print(sample_dict.values())
        print(sample_dict.items())
        print_list(file_list)
#%% 230714
def change_string(input_string):
    t_input = []
    t_input.extend(input_string.replace(' ', '').replace('.', '').replace("'", ''))
    result = '/'.join(t_input)
    return result
           
    
def splitString(text1, symbol1='/'):
    result = symbol1.join(text1.strip().replace('.', '').replace("''", '').split())
    return result
def func_1(a, b):
     '''이 함수는 더하기 함수 입니다.
     인자로는 실수와 정수 두개 다 사용 할 수 있습니다.
     예시 > func_1(1, 2) -> 3'''
     print(a + b)
def add_func(a, b):
     result = a + b
     return result
def no_return_func(a, b, c = 43):
       result = a + b + c
       print(result)
def in_out_gugudan(n, m = ''):
    if m == '':
        for i in range(1, 10):
            print(n, '*', i, '=', n * i)
    else:
        for i in range(n, m+1):
            for j in range(1, 10):
                print(i, '*', j, '=', i * j)
#def gugu(n):
    #for i in range(1, 10):
        #print(n, '*', i, '=', n * i)

#def in_out_gugudan(n, m = ''):
    #if m == '':
        #gugu(n)
    #else:
        #for i in range(n, m+1):
            #gugu(i)
def in_out_gugudan2(num1, num2=0):
    for j in range(num1, num2+1 if num2 !=0 else num1+1):
        print(f'     {j}단')
        for i in range(1,10):
            print(f'{j} X {i} = {j*i}')       # print(f'문자열 {변수} 문자열')
def in_out_gugudan3(n, m = None):
    if m is None:
        m = n+1
    for i in range(n, m):
        for j in range(1, 10):
            print(f' {i} * {j} = {i * j}')
def nothing():
    return
def nothing1():
    return 1, 2, 3
def nothing2():
    pass
def minuss(height, width):     
    return height-width
def varg(a, *arg):
    print(a, arg)
def printf(format, *args):
    print(format % args)
def len_user(random_list):
    num = 0
    for i in random_list:
        if i is not None:
            num += 1
    print(num)
def f(width, height, **kwargs):
    print(width, height)
    print(kwargs)
def g(a, b, *args, **kwargs):      #args : 위치 인수, kwargs : 키워드 인수
    print(a, b)
    print('*args :',args)
    print('**kwargs :', kwargs)
def h(a, b, c):
    print(a, b, c)
def gg(t):
    t = [5,6,7]
def ff(t):
    t[1] = 100000
def n_squared(a, b):
    '''
    a : sequence
    b : integer
    '''
    nsqr_list = []
    for i in a:
        nsqr_list.append(i**b)
    return nsqr_list
def edit_list(sample_list):
    #sample_list[1] = 100
    #dummy_list = copy.copy(sample_list)
    #dummy_list[1] = 100
    dummy_list = copy.deepcopy(sample_list)
    dummy_list[1][0] = 100
    print('subroutine : ', dummy_list)
def gen_edit(sample_list):
    sample_list[1] = 100
    print(f'subroutine :{sample_list}')
def copy_edit(sample_list):
    dummy_list = copy.copy(sample_list)
    dummy_list[1] = 100
    print(f'subroutine : {dummy_list}')
def copy_edit_2nd(sample_list):
    dummy_list = copy.copy(sample_list)
    dummy_list[1][0] = 100
    print(f'subroutine : {dummy_list}')
def deep_copy_edit(sample_list):
    dummy_list = copy.deepcopy(sample_list)
    dummy_list[1][0] = 100
    print(f'subroutine : {dummy_list}')
def no_change_edit(sample_list):
    dummy_list = []
    dummy_list.extend(sample_list)
    dummy_list[1] = 100
    print(f'subroutine : {dummy_list}')
def edit_extend(sample_list):
    dummy_list = []
    dummy_list.extend(sample_list)
    dummy_list[1] = 100

def edit_slicing(sample_list):
    dummy_list = sample_list[:]
    dummy_list[1] = 100

def time_check(sample_list):           #extend, 슬라이싱 속도 비교
    f_check_count = 0
    s_check_count = 0
    for i in range(1, 11):
        check = time.time()
        edit_extend(sample_list)
        f_check = time.time()-check

        #time.sleep(2)                  # 2초 휴식

        check = time.time()
        edit_slicing(sample_list)
        s_check = time.time()-check 
        if f_check > s_check:
            s_check_count += 1
        elif f_check < s_check:
            f_check_count += 1
        
    if s_check_count > f_check_count:
        print('슬라이싱이 평균적으로 더 빠름')
    elif s_check_count < f_check_count:
        print('extend가 평균적으로 더 빠름')
def lotto(n):                           #로또 번호 추출
    lotto_list = list(range(1, 46))
    raffle_list = []
    for i in range(n):
        raffle_number = random.sample(lotto_list, 6)
        raffle_list.append(raffle_number)
        print(random.choice(raffle_list))   # random.choice 랜덤으로 1개 고르기
#%% 230717
def print_list2(sample, n=0):
    for i,j in enumerate(sample, start=1):
        print(i, j)
def create_list(target):
    '''
    리스트 생성 함수
    target : value; int, float, string, any...
    return
        output : list
    creat_list(1234) --> [abcd]
    '''
    dummy_list = []
    dummy_list.append(target)
    return dummy_list
def app_ext_list(sample_list, target):
    if type(target) == list:
        sample_list.extend(target)
    else:
        sample_list.append(target)
    return sample_list
#%% 230717
def create_list(target):
    '''
    리스트 생성 함수
    target : value; int, float, string, any...
    return
        output : list
    creat_list(1234) --> [abcd]
    '''
    dummy_list = []
    dummy_list.append(target)
    return dummy_list
'''
def app_ext_list(sample_list, target):
    if type(target) == list:
        sample_list.extend(target)
    else:
        sample_list.append(target)
    return sample_list
'''
def app_ext_list(sample_list, target):
    sample_list.extend(target)
    sample_list.append(target)
    return sample_list
'''실패한것
def max_min_list(sample_list):
    for i in range(len(sample_list)):
        if sample_list[i] > sample_list[i+1]:
            max_num = sample_list[i]
        elif sample_list[i] < sample_list[i+1]:
            min_num = sample_list[i]
    print(f'minimum : {min_num} \nmaximum : {max_num}')    
'''
def min_list(items):
    minimum = items[0]
    for i in items:
        if i < minimum:
            minimum = 1
    return minimum
def max_list(items):
    maximum = items[0]
    for i in items:
        if i > maximum:
            maximum = i
    return maximum
def create_dict(n):
    '''
    n is numeric
    '''
    sample_dict = {}
    for i in range(1,n+1):
        sample_dict[i] = i**2
    return sample_dict
def add_dict(sample_dict, keys, values):
    for i in keys:
        sample_dict[i] = values[keys.index(i)]
    return sample_dict
def add2_dict(result, keys_list, values_list):
    for i,j in enumerate(keys_list):
        result[j] = values_list[i]
    return result
def cannon_add_dict(result, keys, values):
    if isinstance(keys, list):
        for i in keys:
            result[i] = values[keys.index(i)]
    else:
        result[keys] = values
    return result
def cannon_add_dict2(result, keys, values):
    if isinstance(keys, list):
        add_dict(result,keys, values)
    else:
        result[keys] = values
    return result
'''
x = 10
y = 11
def foo():
     x = 20    # foo 함수의 L에, bar 함수의 E에 해당
     def bar():
        a = 30                  # L에 해당
        print(a, x, y)          # 각 순서대로 L, E, G에 해당
     bar()                      # 30, 20, 11
     x = 40
     bar()                      # 30, 40, 11

foo()

g=10
def f():
    global g
    a = g  
    g = 20
    return a
'''
def outer():
    x = 1                          #L, E
    def inner():
        nonlocal x
        x = 2
        print(f'inner : {x}')
    inner()
    print(f'outer : {x}')
def makeCounter():
    count = 0
    def counter():
        nonlocal count      # 외부 변수 count를 수정하기 위해 nonlocal 사용
        count += 1          
        return count
    return counter
def makeCounter2():
    count = []
    def counter1():
        nonlocal count
        count += [1]  # count = count + [1]
        return count
    return counter1
''' 위와 같음
def makeCounter():
    count = []
    def counter():
        count.append(1)
        return count
    return counter
'''
def quadratic(a, b, c):           #중요함! 클로저 함수
    # This is the outer enclosing function
    cache = {}
    def f(x):
        # This is the nested function
        if x in cache:
            return cache[x]
        y = a*x*x + b*x + c
        cache[x] = y
        return y # = cache[x]
    return f # returns the nested function
def divisor(n):
    divisor_list = []
    for i in range(1, n+1):
        if n % i == 0:
            divisor_list.append(i)
    return divisor_list
def add_(n):
    sums = 0
    for i in range(n+1):
        sums += i
    return sums
def add_recur(n):
    if n == 1:
        return 1
    return n + add_recur(n-1) 
def gcd(a, b):              # 최대공약수 / 유클리드 호제법
    while b != 0:
        a = a % b
        a, b = b, a
    return a
def hanoi_tower(n, source, target, auxiliary):      #하노이탑
    if n > 0:
        # n-1 개의 원반을 보조 기둥으로 옮깁니다.
        hanoi_tower(n-1, source, auxiliary, target)

        # 가장 큰 원반을 목표 기둥으로 옮깁니다.
        print(f"Move disk {n} from {source} to {target}")

        # 보조 기둥에 있는 원반들을 목표 기둥으로 옮깁니다.
        hanoi_tower(n-1, auxiliary, target, source)
#%% 230718
def moves(n, left):
    if n == 0: return
    moves(n-1, not left)
    if left:
        print(f'{n} left')
    else:
        print(f'{n} right')
    moves(n-1, not left)
def prime(n):
    i = 2
    if n == 1:
        print('1은 소수가 아닙니다')
    elif n > 1:
        while n % i != 0:
            i += 1
        if i == n:
            print(f'{n}은 소수입니다')
        else:
            print(f'{n}은 소수가 아닙니다')
    else:
        print('자연수를 입력해주세요')
def prime_number(n):
    if n < 2:
        return False
    if n == 2:
        return True
    else:
        for i in range(2, n):
            if n % i == 0:
                return False
        return True
def find_primes(n):          #n까지의 소수들 구하기
    if n < 2:
        return []
    primes = []
    for num in range(2, n+1):
        if all(num % i != 0 for i in range (2, num)):
            primes.append(num)
    return primes
def nameFes(name, fes):
    fes = fes.copy()
    dict1 = {}
    for i in name:
        r = random.choice(fes)
        dict1[i] = r
        fes.remove(r)
    return dict1
def random_fes(name, fes):
    fes = random.sample(fes, len(name))
    return {n: f for n, f in zip(name, fes)}
'''
def connect(a_list, b_list):
    connect_dict = {}
    while len(connect_dict) != len(a_list):
        connect_dict[random.choice(a_list)] = random.choice(b_list)
    return connect_dict
'''
def random_fes2(name, fes):
    random.shuffle(fes)
    return {n: f for n, f in zip(name, fes)}
def sums(numb, target):
    for i_idx, i in enumerate(numb):
        for j_idx, j in enumerate(numb):
            if i + j == target:
                return i_idx, j_idx
def targetIndex(numb, target):
    for i,j in enumerate(numb):
        a = target - j
        if a in numb:
            return numb.index(i), numb.index(a)      
def find_indexes(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in num_dict:
            return [num_dict[diff], i]
        num_dict[num] = i
    return []
def mysums(*args):
    sums = 0
    for i in range(len(args)):
        sums += args[i]
    return sums
def gamble(base_money, target_money, bet):
    chance = 0
    win = 0
    while base_money < target_money:
        winning = random.randrange(0,2)
        chance += 1
        if  winning == 1:
            base_money += bet
            win += 1
        else:
            base_money -= bet
    odds = float(win / chance)
    return (base_money, chance, odds)

#%%230719
def savePasswd(ids, passwd):
    encrypted1 = hashlib.sha1(passwd.encode()).hexdigest()
    with open('access.text', 'a') as f:                 # 'w'는 덮어쓰기, 'a'는 추가하기
        f.write(ids + ':' + encrypted1 + '\n')          # \n 은 savePasswd를 실행할때마다 다음줄에 추가되게 하기 위함
def checkIfUservalid(ids, passwd):
    import hashlib
    encrypted1 = hashlib.sha1(passwd.encode()).hexdigest()
    ids2_dict = {}
    with open('access.text') as f:
         for line in f:
            ids2, passwd2 = line.strip().split(':')
            ids2_dict[ids2] = passwd2
    if ids2_dict[ids] == encrypted1:
        return True
    return False

#%%230724
def search_file(file_name, keyword):
    result = []
    with open(file_name, 'r', encoding ='utf-8') as f:
        for lines in f:
            if keyword in lines:
                result.append(lines.strip())
        return result

def search_folder(folder, keyword):
    search_result = {}
    for root, dirs, files in os.walk(folder):   #현재 디렉토리, 하위 디렉토리, 파일 이름
        for file in files:
            file_path = os.path.join(root, file)      
            if file.endswith('.txt') or file.endswith('.py'):
                search_result[file_path] = search_file(file_path, keyword)
                if search_result[file_path] == []:
                    del search_result[file_path]
    for file_path in search_result.keys():
        print(file_path)
    path = input('원하는 경로를 입력해주세요 :')
    #output.clear()   #코랩에서만 작동
    #os.system('cls') #스파이더에서 작동안함 -> 파워쉘에서 작동함    
    print(f'경로 : {path}')
    print('='*80)
    for i in search_result[path]:
        print(i)
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
#%%230727
def change_list(filename):
    lines_list = []
    result_list = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        lines_list.append(list(map(int, line.strip().split())))
    for i in range(len(lines_list[0])):
        save_list = []
        for j in lines_list:
            save_list.append(j[i])
        result_list.append(save_list)
    return result_list

def trans_matrix(file_path):
    with open(file_path, 'r') as f:
        matrix = [list(map(int, line.strip().split())) for line in f]
        return [list(row) for row in zip(*matrix)]
    
def change2_list(filename):
    with open(filename) as f:
        lines_list = [list(map(int, line.strip().split())) for line in f.readlines()]
    result_list = [[j[i] for j in lines_list] for i in range(len(lines_list[0]))]
    return result_list
def price_profit(listname):
    if not listname or len(listname) < 2:
        return 0
    min_price = listname[0]
    max_profit = 0 
    for price in listname:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    max_price = min_price + max_profit
    return min_price, max_price
def MarkerMap(data_dict):   #pip install folium 프롬프트로 설치
    maps = folium.Map(location=[37.5602, 126.982], zoom_start=7, tiles='cartodbpositron')
    for i in data_dict:
        name = i
        lat_ = data_dict[i][0]
        long_ = data_dict[i][1]
        folium.CircleMarker([lat_, long_], radius = 4, popup = name, color = 'red', fill_color = 'red').add_to(maps)
    return maps
#%% 계산기
result = 0

def add(num):
    global result
    result += num
    return result

#%% class를 이용한 계산기
class Calculator:
    def __init__(self):     # 생성자,   메서드
        self.result = 0
    def add(self, num):     # 메서드
        self.result += num
        return self.result
#%%
def find_data(dataset):
    key = ['rank', 'movieNm', 'openDt', 'salesAmt']
    result_data = {}
    for i, j in dataset.items():
        if i in key:
            result_data[i] = j
    return result_data

def index_data(data):
    for i in data['boxOfficeResult']['dailyBoxOfficeList']:
        result_data_dict = find_data(i)
        print(result_data_dict)
#%%230728

search = ['REFINE_LOTNO_ADDR', 'REFINE_WGS84_LAT', 'REFINE_WGS84_LOGT', 'OPEN_TM_INFO']     # 위도, 경도, 주소, 오픈시간

def find_data(data_dict):
    result_data_dict = {}
    for i, j in data_dict.items():
        if i in search:
            result_data_dict[i] = j
    return result_data_dict

def MarkerMap(file_name):
    with open(file_name, encoding = 'utf-8') as f:
        datas = json.load(f)
    maps = folium.Map(location=[37.5602, 126.982], zoom_start=7, tiles='cartodbpositron')
    for i in range(len(datas)):
        datas_dict = find_data(datas[i])
        name = datas_dict[search[0]]
        lat_ = float(datas_dict[search[1]]) if datas_dict[search[1]] else None
        long_ = float(datas_dict[search[2]]) if datas_dict[search[2]] else None
        time = datas_dict[search[3]]
        if lat_ is not None and long_ is not None:
            folium.CircleMarker([lat_, long_], radius = 4, popup = (name, time), color = 'red', fill_color = 'red').add_to(maps)
    return maps

search2 = ['RESTRT_NM', 'TASTFDPLC_TELNO', 'REFINE_LOTNO_ADDR', 'REFINE_WGS84_LAT', 'REFINE_WGS84_LOGT']     # 가게 이름, 전화 번호, 주소, 위도, 경도

def find_data2(data_dict):
    result_data_dict = {}
    for i, j in data_dict.items():
        if i in search2:
            result_data_dict[i] = j
    return result_data_dict

def MarkerMap2(file_name):
    with open(file_name, encoding = 'utf-8') as f:
        datas = json.load(f)
    maps = folium.Map(location=[37.5602, 126.982], zoom_start=7, tiles='cartodbpositron')
    for i in range(len(datas)):
        datas_dict = find_data2(datas[i])
        name = datas_dict[search2[0]]
        tel = datas_dict[search2[1]]
        add = datas_dict[search2[2]]
        lat_ = float(datas_dict[search2[3]]) if datas_dict[search2[3]] else None
        long_ = float(datas_dict[search2[4]]) if datas_dict[search2[4]] else None
        if lat_ is not None and long_ is not None:
            folium.CircleMarker([lat_, long_], radius = 4, popup = folium.Popup(f'이름 :{name}, 번호: {tel}, 주소 : {add}', max_width = 300), color = 'red', fill_color = 'red').add_to(maps)
    return maps

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
    
#%%230731

class MyStr:
    def __init__(self, s):
        self.s = s
    def __add__(self, b):
        print('왼쪽에 있을때만 사용해 주세요.')
        return self.s + b
    def __radd__(self, b):
        print('오른쪽도 사용 가능 합니다.')
        return b + self.s

class FourCal2:
    def __init__(self, first, second):
        self.setdata(first, second)
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def __add__(self, a):
        return self.first + self.second + a
    def __radd__(self, a):
        return a + self.first + self.second   
    def __mul__(self, a):
        return self.first * self.second * a
    def __rmul__(self, a):
        return  a * self.first * self.second 
    def __sub__(self, a):
        return self.first - self.second - a
    def __rsub__(self, a):
        return a - self.first - self.second
    def __truediv__(self, a):
        return self.first / self.second / a
    def __rtruediv__(self, a):
        return a / self.first / self.second
    
class A:
    def __repr__(self):
        return('class A print')
class B:
    def __str__(self):
        return('class B print')
class C(A):
    def __str__(self):
        return('class C print')
    
class StringRepr:
    def __repr__(self):
        return 'repr called'

    def __str__(self):
        return 'str called'
    
class StringRepr2:
    def __str__(self):
        return 'str called'
    
class StringRepr3:
    def __repr__(self):
        return 'repr called'
    
class StringRepr4:
    def __init__(self, i = 10):
        self.i = i
    def __repr__(self):
        return 'pm.StringRepr4(100)'
    
class MyClass7:
    def __init__(self, x, y ):
        self.x = x
        self.y = y
    def __repr__(self):
        return f'MyClass(x = {self.x}, y = {self.y})'
    
class Accumulator:
    def __init__(self):
        self.sum = 0
    def __call__(self, *args):
        self.sum += sum(args)
        return self.sum
    
class A:
    def __call__(self, v):
        return v
class B2:
    def func(self, v):
        return v
def check(func):
    if callable(func):
        print('callable')
    else:
        print('not callable')
        
class Factorial:
    def __call__(self, n):
        return self.factorial(n)
    def factorial(self, n):
        result = 1
        if n == 0:
            return result
        else:
            for i in range(1, n+1):
                result *= i
        return result
    
class MyStr2:
    def __init__(self, s):
        self.s = s 
    def __repr__(self):
        return f"My_Str('{self.s}')"
    def __add__(self, n):
        return f"My_Str('{self.s + n}')"
    def __sub__(self, n):
        self_s_list = self.s.split(' ')
        num = self_s_list.index(n)
        return MyStr2(' '.join(self_s_list[:num] + self_s_list[num+1:]))
    
class Test_no_getitem:
    def __init__(self):
        print('생성자 __init__을 호출하였습니다.')
        self._numbers = [ n for n in range(1, 11)]
    def __getitem__(self, index):
        print('__getitem__을 호출하였습니다.')
        return self._numbers[index]
    
class MyList:
    def __init__(self, items):
        self.items = items
    def __getitem__(self, x):
        if isinstance(x, int):
            return self.items[x]
        elif isinstance(x, str):
            return self.items.index(x)
        else:
            raise TypeError('잘못된 형식입니다.')
            
class Square:
    def __init__(self, items):
        self.items = [ i**2 for i in range(items+1) ]
    def __getitem__(self, indexs):                      # 반복자 for문 사용이나 형변환시 오류발생 고려
        if len(self.items) >= indexs >= 0:
            return self.items[indexs]
        raise IndexError('list index out of range')
        
class Person:
    def __init__(self, name, phone=None):
        self.name = name
        self.phone = phone
    def __repr__(self):
        return f'<Person {self.name} {self.phone} >'
class Employee(Person):
    def __init__(self, name, phone, position, salary):
        #Person.__init__(self, name, phone)
        #super(Employee, self).__init__(name, phone)
        super().__init__(name, phone)

        self.position = position
        self.salary = salary
        
class Employee2(Person):
    def __init__(self, name, phone, position, salary):
        super().__init__(name, phone)
        self.position = position
        self.salary = salary
    def __repr__(self):
        s = super().__repr__()
        return s + f'Employee {self.name} {self.phone} {self.position} {self.salary}'
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def area(self):
        return 0
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    def __repr__(self):
        return f'x = {self.x}, y = {self.y}'
class Circle(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.radius = r
    def area(self):
        return math.pi * self.radius * self.radius
    def __repr__(self):
        return f'{super().__repr__()}, radius = {self.radius}'
class Cylinder(Circle):
    def __init__(self, x, y, r, h):
        super().__init__(x, y, r)
        self.height = h
    def area(self):
        return 2 *Circle.area(self) + 2*math.pi*self.radius * self.height
    def volum(self):
        return super().area()*self.height
    def __repr__(self):
        return f'{super().__repr__()} height={self.height}'   
    
class Job:
    def __init__(self, position, salary):
        self.position = position
        self.salary = salary
    def __repr__(self):
        return f'Job position = {self.position} salary = {self.salary}'
class Employee3(Person, Job):
    def __init__(self, name, phone, position, salary):
        Person.__init__(self, name, phone)
        Job.__init__(self, position, salary)
        
    def __repr__(self):
        return Person.__repr__(self) + Job.__repr__(self)    

#%%230801
class A:
    def __init__(self, x):
        self.x = x
    def print_x(self):
        print(self.x)
class B(A):
    def setdata(self, y):
        self.y = y
    def print_y(self):
        print(self.y)
        
class Person:
    def __init__(self, name, phone=None):
        self.name = name
        self.phone = phone
    def __repr__(self):
        return f'<Person {self.name} {self.phone} >'
class Employee(Person):
    def setdata(self, position, salary):
        self.position = position
        self.salary = salary
    def print_info(self):
        print(self.name, self.phone)

class Stack(list):
    push = list.append

class Queue(list):
    enqueue = list.append
    def dequeue(self):
        return self.pop(0)

class MyDict(dict):
    def keys(self):
        L = super().keys()
        return sorted(L)
    
class Animal:
    def cry(self):
        print('...')
class Dog(Animal):
    def cry(self):
        print('멍멍')
class Duck(Animal):
    def cry(self):
        print('꽥꽥')
class Fish(Animal):
    pass

class OrderedList(list):
    def __init__(self, items):
        super().__init__(items)
        self.sort()
    def append(self, items):
        super().append(items)
        self.sort()
    def extend(self, items):
        super().extend(items)
        self.sort()
    
class Counter:
    def __init__(self, n = 0, step = 1):
        self.n = n
        self.step = step
    def incr(self):
        self.n += self.step
        return self.n
    def __repr__(self):
        return str(self.n)
    def __str__(self):
        return str(self.n)
    def __call__(self):
        return self.incr()

class Book:
    def set_info(self, title, name):
        self.title = title
        self.name = name
    def print_info(self):
        print(f'책 제목: {self.title}\n책 저자: {self.name}\n\n')

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
#%%230823
def creatArray(n):
    if 1 <= n <= 30:
        if n == 1:
            a = [[1]]
        else:
            a = [[0 for i in range(n)] for j in range(n)]
            x, y = 0, 0
            vec = 'right'
            for num in range(1, n**2+1):
                a[x][y] = num
                if vec == 'right':
                    if y < n-1 and a[x][y+1] == 0:
                        y += 1
                    elif a[x+1][y] == 0:
                        vec = 'down'
                        x += 1
                elif vec == 'down':
                    if x < n-1 and a[x+1][y] == 0:
                        x += 1
                    elif a[x][y-1] == 0:
                        vec = 'left'
                        y -= 1
                elif vec == 'left':
                    if y > 0 and a[x][y-1] == 0:
                        y -= 1
                    elif a[x-1][y] == 0:
                        vec = 'up'
                        x -= 1
                elif vec == 'up':
                    if x > 0 and a[x-1][y] == 0:
                        x -= 1
                    elif a[x][y+1] == 0:
                        vec = 'right'
                        y += 1
        return a
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.dot(x,w) + b
    if tmp < 0 :
        return 0
    else:
        return 1
    
def step_func(x):
    if x>0:
        return 1
    else:
        return 0

def aryAsbool(x):
    if isinstance(x, np.ndarray):
        y = x>0
        y = y.astype(int)
        return y
    elif isinstance(x, list):
        x = np.array(x)
        return aryAsbool(x)
    else:
        print('list 혹은 array를 입력해주세요')
        
def Plotbool(x):
    plt.plot(x, aryAsbool(x))
    return plt.show()
def sigmoid(x):
    if isinstance(x, np.ndarray):
        return 1/(1+np.exp(-x))
    elif isinstance(x, list):
        x = np.array(x)
        return sigmoid(x)

def Plotsigmoid(x):
    if isinstance(x, np.ndarray):
        plt.plot(x, sigmoid(x))
        return plt.show()
    else:
        print('array를 삽입해주세요!')
        
def Plotsigbool(x):
    if isinstance(x, np.ndarray):
        y1 = sigmoid(x)
        y2 = aryAsbool(x)
        plt.plot(x, y1, label='sigmoid')
        plt.plot(x, y2, linestyle='--', label='bool')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(-0.1, 1.1)
        return plt.show()
    else:
        print('array를 삽입해주세요')
        
def relu(x):
    return np.maximum(0, x)

def Plotrelu(x):
    if isinstance(x, np.ndarray):
        plt.plot(x, relu(x))
        return plt.show()
    else:
        print('array를 삽입해주세요')

def straightPlot(x):
    if isinstance(x, np.ndarray):
        y = -x + 0.5
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        plt.axvline(x=0,color='k')
        plt.axhline(y=0,color='k')
        plt.scatter([0],[0],marker='o',color='r')
        plt.scatter([1,0,1],[0,1,1],marker='^',color='r')
        plt.xlabel("x") # x축 이름
        plt.ylabel("y") # y축 이름
        plt.fill_between(x,y,-2, alpha=0.5) # y값에서 y축의 -3 까지의 값들에 대해 색깔칠해준다.
        plt.grid()
        return plt.show()
    elif isinstance(x, list):
        y = x[:]
        for idx, i in enumerate(y):
            y[idx] = -i + 0.5
        plt.figure(figsize=(8,6))
        plt.plot(x, y)
        plt.axvline(x=0,color='k')
        plt.axhline(y=0,color='k')
        plt.scatter([0],[0],marker='o',color='r')
        plt.scatter([1,0,1],[0,1,1],marker='^',color='r')
        plt.xlabel("x") # x축 이름
        plt.ylabel("y") # y축 이름
        plt.fill_between(x,y,-2, alpha=0.5) # y값에서 y축의 -3 까지의 값들에 대해 색깔칠해준다.
        plt.grid()
        return plt.show()
    else:
        print('array 혹은 list를 삽입해주세요')
def getDat(path):
    df = pd.read_table(path, header=None, sep='::')
    return df

def getDummies(df):        
    col = []
    gen_list = []
    new_df = df.loc[:, :'title']
    
    for genres in df['genres']:
        gen_list.extend(genres.split('|'))
    gen_list = list(set(gen_list))
    gen_list.sort()
    
    for idx1 in range(len(df.index)):
        col.append(pd.get_dummies(df.loc[idx1,'genres']).columns[0].split('|'))
    
    for idx2, j in enumerate(col):
        for gen in gen_list:
            if gen in j:
                new_df.loc[idx2,f'Genres_{gen}'] = 1
            else:
                new_df.loc[idx2,f'Genres_{gen}'] = 0
    return new_df

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
#%%230829

def delSpace(val):
    return val.replace(' ', '')

def delComma(val):
    return val.replace(',', '')

def dfPop(df_pop):
    df_pop.drop(index=[0,1], inplace=True)      
    df_pop.rename(columns={'동별(2)':'구분', '2023 1/4.1':'총 인구수'}, inplace=True)
    df_pop_result = df_pop[['구분', '총 인구수']].copy()
    df_pop_result['총 인구수'] = df_pop_result['총 인구수'].astype(int)
    return df_pop_result

def dfCctv(df_cctv):
    df_cctv.drop(index=0, inplace=True)
    df_cctv.rename(columns={'총계':'CCTV 갯수'}, inplace=True)
    df_cctv['구분'] = df_cctv['구분'].apply(delSpace)
    df_cctv['CCTV 갯수'] = df_cctv['CCTV 갯수'].apply(delComma)
    df_cctv_result = df_cctv[['구분', 'CCTV 갯수']].copy()
    df_cctv_result['CCTV 갯수'] = df_cctv_result['CCTV 갯수'].astype(int)
    return df_cctv_result

def dfMerge(x_df, y_df):
    x_df2 = dfPop(x_df)
    y_df2 = dfCctv(y_df)
    return pd.merge(x_df2, y_df2, on ='구분')


def dfPlotScatt(new_df, font_path):
    new_df.rename(columns={'CCTV 갯수':'cctv_갯수', '총 인구수':'총_인구수'}, inplace=True)
    x = new_df.총_인구수.values.reshape(-1,1)
    y = new_df.cctv_갯수
    model = LinearRegression()
    model.fit(x,y)
    new_df['predict_lr'] = model.predict(x)
    
    new_df['res'] = new_df.cctv_갯수 - new_df.predict_lr
    new_df_sort = new_df.sort_values(by='res', ascending=False)
    
    fontprop = font_manager.FontProperties(fname=font_path, size=20)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    scatter = ax.scatter(new_df_sort.총_인구수, new_df_sort.cctv_갯수, c = new_df_sort.res, s=50, label = 'cctv')
    ax.plot(new_df_sort.총_인구수, new_df_sort.predict_lr, color = 'r', label = 'predict')
    
    for n in new_df_sort.index:
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, 'cctv_갯수'], new_df_sort.loc[n, '구분'], fontproperties=fontprop, size =10 )
    plt.colorbar(scatter, ax=ax)
    ax.legend()
    ax.set_xlabel('총 인구수', fontproperties=fontprop)
    ax.set_ylabel('cctv 갯수', fontproperties=fontprop)
    plt.show()
    
#%%230829
def selectBorough(borough):
    if '서울' in borough:
        return borough.split(' ')[1][:-1]
    elif len(borough.split(' ')) == 1:
        return borough[:-1]
    else:
        return borough.split(' ')[0]
    
def setCCTV(df):
    df1 = df.groupby(df['관리기관명'], as_index=False).sum().loc[:,['관리기관명', '카메라대수']].copy()
    df1['관리기관명'] = df1['관리기관명'].apply(selectBorough)
    df_cctv = df1
    return df_cctv

def setPOP(df):
    df_pop = df.copy()
    df_pop.rename(columns={'동별(2)':'관리기관명', '2023 1/4.1':'총_인구수'}, inplace=True)
    df_pop = df_pop[['관리기관명','총_인구수']]
    df_pop.drop(index=[0,1], inplace=True)
    return df_pop

def mergedf(df1, df2):
    new_df = pd.merge(setCCTV(df1), setPOP(df2))
    new_df['총_인구수'].values
    new_df['총_인구수'] = new_df['총_인구수'].astype(int)
    return new_df

def LRdf(df1, df2):
    new_df = mergedf(df1, df2)
    x = new_df['총_인구수'].values.reshape(-1,1)
    y = new_df['카메라대수']
    model = LinearRegression()
    model.fit(x,y)
    new_df['predict_lr'] = model.predict(x)
    new_df['residual'] = new_df['카메라대수'] - new_df['predict_lr']
    new_df_sort = new_df.sort_values(by='residual', ascending=False)
    new_df_sort.reset_index(inplace=True)
    new_df_sort.drop(columns='index', inplace=True)
    return new_df_sort

def PredictModelPlot(df1, df2):
    new_df_sort = LRdf(df1, df2)
    
    font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'
    fontprop = font_manager.FontProperties(fname=font_path, size=20)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    scatter = ax.scatter(new_df_sort['총_인구수'], new_df_sort['카메라대수'], c=new_df_sort['residual'], label='자치구')
    ax.plot(new_df_sort['총_인구수'], new_df_sort['predict_lr'], c='r', label='예측값')
    ax.legend(prop=fontprop)
    ax.set_xlabel('총 인구수', fontproperties=fontprop)
    ax.set_ylabel('카메라대수', fontproperties=fontprop)
    plt.colorbar(scatter, ax=ax)
    for n in range(3):
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, '카메라대수']*1.01,
                new_df_sort.loc[n, '관리기관명'], fontproperties=fontprop, size=10)
    for n in range(21, 24):
        ax.text(new_df_sort.loc[n, '총_인구수']*1.02, new_df_sort.loc[n, '카메라대수']*1.01,
                new_df_sort.loc[n, '관리기관명'], fontproperties=fontprop, size=10)
    plt.show()


#%%230831
def findCodes(name):
    code_all = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download')
    code_df = code_all[0]
    expr_str = f'''회사명 == '{name}' '''
    code = code_df.query(expr_str)
    target_code = '{:06d}'.format(code['종목코드'].values[0])
    return target_code
    
def devTrainTest(df, column, date):
    df[column] = df[column].astype(int)
    train_df = df.query(f'''index <= '{date}' ''')
    test_df = df.query(f'''index > '{date}' ''')
    return train_df, test_df

def prePlot(df, column, date, degrees=10):
    train_df, test_df = devTrainTest(df, column, date)
    
    polynomial_features = PF(degree=degrees, include_bias=False)
    linear_regression = LR()
    pipeline = Pipeline( [ ('polynomial_features', polynomial_features), ('linear_regression', linear_regression) ] )
    x = np.arange(len(train_df.index)).reshape(-1,1)
    y = train_df[column]
    pipeline.fit(x, y)
   
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 1, 1)
    plt.plot(x, pipeline.predict(x), label='Model', color='red')
    plt.scatter(x, y, edgecolors='b', s=5, label='Samples', color='skyblue')
    plt.title(f'Degree=degrees')
    plt.legend(loc='best')
    plt.show()

def prophetPlot(df, column, date, period):
    train_df, test_df = devTrainTest(df, column, date)
    
    df_prophet = train_df[column]
    df2 = df_prophet.reset_index()
    df2.columns = ['ds', 'y']
    m = Prophet()
    m.fit(df2)
    future = m.make_future_dataframe(periods=period)
    forcast = m.predict(future)
    m.plot(forcast)
    plt.show()
    
def toDateTime(val):
    return datetime.strptime(val, '%Y-%m-%d %H:%M:%S')

def timeDfAgg(df, column1, column2, *args):
    if type(df[column1][0]) != 'pandas._libs.tslibs.timestamps.Timestamp':
        if df[column2].isna().any():
            df.dropna(inplace=True)           
        df[column1] = df[column1].apply(toDateTime)
        df.set_index(column1, inplace=True)
        df_re = df.resample('d')
        result_df = df_re.agg(args)
    else:
        if df[column2].isna().any():
            df.dropna(inpace=True)
        df_re = df.resample('d')
        result_df = df_re.agg(args)

    return result_df

#230904
def delHyphen(val):  #   하이픈 제거
    if isinstance(val, str) and '-' in val:
        val = np.nan
    else:
        val = val
    return val

def delTr(val): #   Tr, tr 제거 -> 없거나 정말 미세한 양
    if val == 'Tr':
        val = np.nan
    elif val == 'tr':
        val = np.nan
    else:
        val = val
    return val

def delBracket(val):    #   괄호 제거
    if type(val) == str:
        if val.startswith('('):
            val = np.nan
        else:
            val = val
    else:
        val = val
    return val

# 단위 쓰기 쉽게 변경
def changeColumn(column):
    if '㎎' in column:
        column = column.replace('㎎', 'micro')
    elif '㎍' in column:
        column = column.replace('㎍', 'micro')
    elif 'g' in column:
        column = column.replace('g', 'micro')
    return column

    return val

# 영양소 단위 통일
def changeWei(df, column):
    if 'g' in column:
        df[column] = df[column] * 1000000
    elif '㎎' in column:
        df[column] = df[column] * 1000
    return df

# 전처리
def refine(df):
    for column in df.columns:
        df[column] = df[column].apply(delHyphen)
        df[column] = df[column].apply(delBracket)
        df[column] = df[column].apply(delTr)

    for column in df.columns:
        if '단백질' in column:
            start = list(df.columns).index(column)
        elif '회분' in column:
            end = list(df.columns).index(column)

    for col in df.columns[start:end+1]:
        df[col] = df[col].astype(float)
        df = changeWei(df, col)
        new_column = changeColumn(col)
        df.rename(columns={col : new_column}, inplace=True)
    return df

#%%230905

def readweb(rows):
    etfs = {}
    for row in rows:
        for index in range(len(row.text.split('\n'))):

            try:
                row_text = row.text.split('\n')[index]
                start = row_text.rfind('(')
                end = row_text.rfind(')')
                if start > 2:
                    new_row = unicodedata.normalize("NFKD", row_text)
                    if '|' in new_row[start+1:end].replace(':', ''):
                        etf_name = [new_row[:start-1]]
                        etf_market = [new_row[start+1:end].replace(':', '').split('|')[0]]
                        etf_ticker = [new_row[start+1:end].replace(':', '').split('|')[-1]]
                    else:
                        etf_name = [new_row[:start-1]]
                        etf_market = [' '.join(new_row[start+1:end].replace(':', '').split(' ')[:-1])]
                        etf_ticker = [new_row[start+1:end].replace(':', '').split(' ')[-1]]
                        
                    if (len(etf_ticker) > 0) & (len(etf_market) > 0) & (len(etf_name) > 0):
                        etfs[etf_ticker[0]] = [etf_market[0], etf_name[0]]            
            except AttributeError as err:
                pass
    return etfs

#%%231005

def split_data(dict_, x_key, y_key, size=0.2, state=2):
    dict_copy = dict_.copy()
    x_train, x_test, y_train, y_test = train_test_split(dict_copy[x_key], dict_copy[y_key], test_size=size, random_state=state)
    return x_train, x_test, y_train, y_test

def DT(x_train, y_train, x_test, state=2):
    dt_clf = DecisionTreeClassifier(random_state=state)
    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test) # 만들어진 모델에 x_test값 넣어서 예측값 도출하기
    return pred

def acc(y_test, pred):
    return accuracy_score(y_test, pred)

#%%231006

#랜덤포레스트
def RFC(df, target, size=0.2, state=2):	# target : 종속변수
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=size, random_state=state)
    rf_clf = RandomForestClassifier(random_state=state)
    rf_clf.fit(x_train, y_train)
    pred = rf_clf.predict(x_test)
    acc = accuracy_score(y_test, pred)
    return acc

#KFold
def KFd(df, target, n_split=3, state=2):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    kfold = KFold(n_splits=n_split)
    cv_acc = []
    df_clf = DecisionTreeClassifier(random_state=state)
    for train_index, test_index in kfold.split(x_):
        df_clf.fit(x_.iloc[train_index], y_.iloc[train_index])
        pred = df_clf.predict(x_.iloc[test_index])
        cv_acc.append(accuracy_score(y_.iloc[test_index], pred))
    return np.mean(cv_acc)

#StratifiedKFold
def SKF(df, target, n_split=3, state=2):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    cv_acc = []
    skf = StratifiedKFold(n_splits=n_split)
    df_clf = DecisionTreeClassifier(random_state=state)
    for train_index, test_index in skf.split(x_, y_):
        df_clf.fit(x_.iloc[train_index], y_.iloc[train_index])
        pred = df_clf.predict(x_.iloc[test_index])
        cv_acc.append(accuracy_score(y_.iloc[test_index], pred))
    return np.mean(cv_acc)

#cross_val_score
def CVS(df, target, cv=3):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    dt_clf = DecisionTreeClassifier()
    scores = cross_val_score(dt_clf, x_, y_, scoring='accuracy', cv=cv)
    return scores

#GridSearchCV
def GS(df, target, size=0.2, state=2, max_depths=[1, 2, 3], min_samples_splits=[2,3], cv=3):
    x_ = df.drop(target, axis=1)
    y_ = df[target]
    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=size, random_state=state)
    dt_clf = DecisionTreeClassifier()
    params={'max_depth':max_depths, 'min_samples_split':min_samples_splits}
    grid_dt_clf = GridSearchCV(dt_clf, param_grid=params, cv=cv, refit=True)
    grid_dt_clf.fit(x_train, y_train)
    new_df = pd.DataFrame(grid_dt_clf.cv_results_)
    return new_df[['params', 'mean_test_score', 'rank_test_score']]

#%%231012

def grid_cv(X, y, model, parameters):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=156)
    grid = GridSearchCV(model, param_grid=parameters, cv=5)
    grid.fit(X_train, y_train)
    pred = grid.best_estimator_.predict(X_test)
    print('사용한 모형 :', model)
    print('적합한 모델 :', grid.best_estimator_)
    print('best_score :', np.round(grid.best_score_, 4))
    print('best_params :', np.round(grid.best_params_, 4))
    print('accuracy :', np.round(accuracy_score(y_test, pred), 4))

#%% 231016

def repair(df):
    df['windspeed'].fillna((df['windspeed'].median()), inplace=True)
    df['hum'].fillna(df.groupby(['season'])['hum'].transform('median'), inplace=True)
    temp_mean = (df.iloc[700]['temp'] + df.iloc[702]['temp']) / 2
    atemp_mean = (df.iloc[700]['atemp'] + df.iloc[702]['atemp']) / 2
    df['temp'].fillna(temp_mean, inplace=True)
    df['atemp'].fillna(atemp_mean, inplace=True)
    df['dteday'] = df['dteday'].apply(pd.to_datetime, infer_datetime_format=True, errors='coerce')
    df['mnth'] = df['dteday'].dt.month
    df.loc[730, 'yr'] = 1.0
    df.drop(['casual', 'registered'], axis=1, inplace=True)
    df.drop(['instant', 'dteday'], axis=1, inplace=True)
    return df

def get_lr(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return mse**0.5

def get_xg(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    xg_reg = XGBRegressor()
    xg_reg.fit(X_train, y_train)
    pred = xg_reg.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return np.sqrt(mse)

def cross_score(model, X, y):
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
    return np.sqrt(-scores)

def gbm_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    tree1 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree1.fit(X_train, y_train)
    y_train_pred = tree1.predict(X_train)
    y2_train = y_train - y_train_pred

    tree2 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree2.fit(X_train, y2_train)
    y2_train_pred = tree2.predict(X_train)
    y3_train = y2_train - y2_train_pred

    tree3 = DecisionTreeRegressor(max_depth=2, random_state=2)
    tree3.fit(X_train, y3_train)
    
    y1_pred = tree1.predict(X_test)
    y2_pred = tree2.predict(X_test)
    y3_pred = tree3.predict(X_test)
    y_pred = y1_pred + y2_pred + y3_pred

    return np.sqrt(mean_squared_error(y_test, y_pred))

def GBM(X, y, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    gbr = GradientBoostingRegressor(max_depth=2, n_estimators=300, random_state=2, learning_rate=learning_rate)
    gbr.fit(X_train, y_train)
    pred = gbr.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))

def plot_gbm_rmse(rmse_list, x=None, values='learning_rate'):
    if x is None:
        x = np.arange(len(rmse_list))
    plt.plot(x, rmse_list)
    plt.xlabel(values)
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()

def GBM_depth(X, y, depths):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=156)
    rmse_dict = {}
    for depth in depths:
        gbr = GradientBoostingRegressor(max_depth=depth, n_estimators=300, random_state=156, learning_rate=0.2)
        gbr.fit(X_train, y_train)
        pred = gbr.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        rmse_dict[rmse] = depth
    return rmse_dict

#%% 231018

def grid_search(params, X, y, model=XGBClassifier(), cv=5, scoring='accuracy', random=False):
    clf = model
    if random:
        grid = RandomizedSearchCV(clf, params, cv=cv, n_jobs=-1, scroring=scoring, random_state=2)
    else:
        grid = GridSearchCV(clf, params, cv=cv, n_jobs=-1, scoring=scoring)
    grid.fit(X, y)
    print(f'best_params : {grid.best_params_}')
    print(f'best_score : {grid.best_score_}')

def early_end(X, y, model=XGBClassifier(), rounds=10, vb=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
    clf = model
    clf.fit(X_train, y_train, eval_set= [(X_test, y_test)], eval_metric='error', early_stopping_rounds=rounds, verbose=vb)
    pred = clf.predict(X_test)
    return accuracy_score(y_test, pred)

def atlas_(df):
    X=df.loc[:, ~df.columns.isin(['EventId', 'Label', 'Weight'])]
    y = df.loc[:, 'Label']
    s = np.sum(df[df.Label==1])['test_Weight']
    b = np.sum(df[df.Label==0])['test_Weight']
    clf = XGBClassifier(n_estimators=120, learning_rate=0.1, missing=-999.0, scale_pos_weight=b/s)
    clf.fit(X, y, sample_weight=df['test_Weight'], eval_set=[(X, y)], eval_metric=['auc', 'ams@0.15'], sample_weight_eval_set=[df['test_Weight']])
    print(clf.evals_result())

def to_week(val):
    if val in [5, 6]:
        return 1
    else:
        return 0

def rush_hour(row, column1, column2):
    if (row[column1] in [6, 7, 8, 9, 15, 16, 17, 18]) and (row[column2] == 0):
        return 1
    else:
        return 0

#%%231020

def svc_plot(X, y):
    clf = SVC(kernel = 'linear')
    clf.fit(X, y)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    a= -w[0]/w[1]
    b = -b/w[1]
    half_margin = 1/w[1]
    x_data = np.arange(X.max()+2)
    print(x_data)
    hyperline = a*x_data + b
    up_line = hyperline + half_margin
    down_line = hyperline - half_margin

    plt.figure(figsize=(6,4))
    plt.plot(X[:, 0][y==y[0]], X[:, 1][y==y[0]], 'rx')
    plt.plot(X[:, 0][y!=y[0]], X[:, 1][y!=y[0]], 'bo')
    plt.plot(x_data, hyperline)
    plt.plot(x_data, up_line, '--', c='green')
    plt.plot(x_data, down_line, '--', c='green')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.show()

#def cancer_svc_plot(X, y):
    

'''
#%%231024

def get_best_degree(model, degrees, X, y):
    result = {}
    for deg in degrees:
        pipe = Pipeline([(PolynomialFeatures().__class__.__name__, PolynomialFeatures(degree=deg, include_bias=False)), (model.__class__.__name__, model)])
        scores = cross_val_score(pipe, X, y, scoring='neg_mean_squared_error', cv=10)
        pipe.fit(X, y)
        result[deg] = (pipe, -scores.mean(), scores.std())
    
    mses = {}
    for i in result:
        #w = result[i][0].named_steps[model.__class__.__name__].coef_
        mses[i] = result[i][1]

    best_degree = min(mses, key=mses.get)

    X_test = np.linspace(0, 1, 100)
    y_test = true_func(X_test)
    X_test = X_test.reshape(-1, 1)
    plt.figure(figsize=(14,5))
    num=0
    for i in result:
        pipelines = result[i][0]
        mean_score = result[i][1]
        std_score = result[i][2]
        
        pred = pipelines.predict(X_test)
        num+=1
        ax = plt.subplot(1, len(degrees), num)
        plt.plot(X_test, pred, label = 'model')
        plt.plot(X, y, 'ko', label='samples')
        plt.plot(X_test, y_test, label = 'True function')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y', rotation=0)
        plt.title(f'Degree : {i}\nMSE : {mean_score}')
        plt.setp(ax, xticks=(), yticks=())
    plt.show()

    return best_degree
        
def true_func(X):
    return np.cos(1.5 * np.pi * X)
'''

#%%231025

def get_linear_reg_eval(model_name, params=None, X=None, y=None):
    coef_df = pd.DataFrame()
    for a in params:
        if model_name == 'Ridge':
            model = Ridge(alpha=a)
        if model_name == 'Lasso':
            model = Lasso(alpha=a)
        if model_name == 'ElasticNet':
            model = ElasticNet(alpha=a, l1_ratio=0.7)

        model.fit(X, y)
        coef_sr = pd.Series(model.coef_, index=X.columns)
        column_name = 'alpha:'+str(a)
        coef_df[column_name] = coef_sr
        neg_mse_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)
        rmse = np.sqrt(-neg_mse_score)
        avg_rmse = np.mean(rmse)
        print(f'== alpha : {a} ==\navg_rmse : {np.round(avg_rmse, 4)}')
    return coef_df
 
def rm_outlier(x, y):
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    fence_low = q1-1.5*iqr
    fence_high = q3+1.4*iqr
    outlier = []
    for i in y:
        if i<fence_low or i>fence_high:
            outlier.append(i)
    print('outlier : ', np.round((len(outlier)/y.shape[0])*100, 4))
    x = x[(y>fence_low)&(y<fence_high)]
    y = y[(y>fence_low)&(y<fence_high)]
    return x, y

def visualize_target(y):
    fig = plt.figure(constrained_layout=True, figsize=(12,6))
    grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)

    ax1 = fig.add_subplot(grid[0:2, :4])
    ax1.set_title('Histogram')
    sns.distplot(y, norm_hist=True,fit=norm, ax = ax1,color='indianred')

    ax2 = fig.add_subplot(grid[2:, :4])
    ax2.set_title('QQ_plot')
    stats.probplot(y, plot = ax2)

    ax3 = fig.add_subplot(grid[:, 4])
    ax3.set_title('Box Plot')
    sns.boxplot(y=y, orient='v', ax = ax3,color='indianred')
    plt.show()

def Regression_process(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=156)

    lr = LinearRegression().fit(X_train, y_train)
    print(f'R2 (train set, test set) : {np.round(lr.score(X_train, y_train), 4)}, {np.round(lr.score(X_test, y_test), 4)}')

def data_process(X, y):
    X, y = rm_outlier(X, y)
    visualize_target(y)
    Regression_process(X, y)
    lr = LinearRegression()
    mse = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse = np.sqrt(-mse)
    avg_rmse = np.mean(rmse)
    print(f'LinearRegression_avg_rmse : {avg_rmse}')
    ridge=Ridge(alpha=10)
    mse = cross_val_score(ridge, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse = np.sqrt(-mse)
    avg_rmse = np.mean(rmse)
    print(f'Ridge_avg_rmse : {avg_rmse}')
    return X, y

#%%231026
def get_frc(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print(f'TN, FP : {np.round(confusion, 4)[0]}')
    print(f'FN, TP : {np.round(confusion, 4)[1]}')
    print(f'accuracy : {accuracy}')
    print(f'precision : {precision}')
    print(f'recall : {recall}')
    print(f'f1 : {f1}')
    print(f'roc_auc : {roc_auc}')

def get_pred_predproba(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], stratify=df.iloc[:, -1], test_size=0.2, random_state=156)
    logr = LogisticRegression()
    logr.fit(X_train, y_train)
    pred = logr.predict(X_test)
    pred_proba = logr.predict_proba(X_test)[:, 1]
    return y_test, pred, pred_proba

def case_get(df):
    cases = ['case1', 'case2', 'case3']
    for case_ in cases:
        df_copy = df.copy()
        if case_ == 'case1':
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount 변환 x ======')
            get_frc(y_test, pred, pred_proba)
        if case_ == 'case2':
            scaler = StandardScaler()
            Amount_scaled = scaler.fit_transform(np.array(df_copy.Amount).reshape(-1, 1))
            df_copy['Amount'] = Amount_scaled
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount use StandardScaler ======')
            get_frc(y_test, pred, pred_proba)
        if case_ == 'case3':
            Amount_scaled = np.log1p(np.array(df_copy.Amount))
            df_copy['Amount'] = Amount_scaled
            y_test, pred, pred_proba = get_pred_predproba(df_copy)
            print('====== Amount use log1p ======')
            get_frc(y_test, pred, pred_proba)

#%%231030

def get_model_down_dim(model, X_features, y_target):
    pca = PCA(n_components=2)
    pca.fit(X_features)
    X_features_pca = pca.transform(X_features)
    X_train, X_test, y_train, y_test = train_test_split(X_features_pca, y_target, random_state=2)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = model
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f'{clf.__class__.__name__} 의 정확도 : {accuracy_score(y_test, pred)}')
    return clf

#%%231031

def cluster_(df, target_name, pca=True):
    if pca:
        component = int(input('n_components를 입력하세요: '))
        pca_ = PCA(n_components=component)
        pca_transformed = pca_.fit_transform(df.iloc[:, :-1])
        df['pca_x'] = pca_transformed[:, 0]
        df['pca_y'] = pca_transformed[:, 1]
        
        kmeans = KMeans(n_clusters=len(df[target_name].unique()), init='k-means++', max_iter=500, random_state=0)
        kmeans.fit(df.loc[:, ['pca_x', 'pca_y']])
        df['pca_cluster'] = kmeans.labels_
        print(df.groupby(['target', 'pca_cluster']))
        markers=['s', 'x', 'o']
        for i in range(len(df[target_name].unique())):
            plt.scatter(df[df.pca_cluster==i].pca_x, df[df.pca_cluster==i].pca_y, marker=markers[i])
        plt.title('PCA_CLUSTER')
        return plt
    else:
        kmeans = KMeans(n_clusters=len(df[target_name].unique()), init='k-means++', max_iter=500, random_state=0)
        kmeans.fit(df.iloc[:, :-1])
        df['cluster'] = kmeans.labels_
        print(df.groupby(['target', 'cluster']).count())
        markers=['s', 'x', 'o']
        for i in range(len(df[target_name].unique())):
            plt.scatter(df[df.cluster==i].pca_x, df[df.cluster==i].pca_y, marker=markers[i])
        plt.title('CLUSTER')
        return plt

def cust_cluster(df):
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=1000, random_state=0)
    labels = kmeans.fit_predict(df.values)

#%%231101
nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('all')
def token_text(text, stopwords_list):
    sentences = sent_tokenize(text)
    word_list = []
    remove_dict = {}
    for i in string.punctuation:
        remove_dict[ord(i)] = None
    for sentence in sentences:
        sentence = sentence.translate(remove_dict)
        words = word_tokenize(sentence)
        token_list = []
        for word in words:
            word = word.lower()
            if word not in stopwords_list and len(word) > 1:
                token_list.append(word)
        word_list.append(token_list)
    return word_list
#%%231102
nltk.download('all')
def LemTokens(tokens):
    lemmar = WordNetLemmatizer()
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def cluster_text(document_df):
    tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english', ngram_range=(1,2), max_df = 0.85, min_df=0.05)

    feature_vect = tfidf_vect.fit_transform(document_df.opinion_text)
    km_cluster = KMeans(n_clusters=3, max_iter=10000, random_state=0)
    cluster_label = km_cluster.fit_predict(feature_vect)
    document_df['cluster_label'] = cluster_label
    return document_df

#%%231103 

#활성화 함수
def step_func(x):
    y = x>0
    return y.astype('int')

def sigmoid(x): # or scipy.special.expit
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_func(x):
    return x

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#%%231109

def model_fn(X_data, y_target, optimizer_, epochs, dropout=None):
    train_scaled, val_scaled, train_target, val_target = train_test_split(X_data, y_target, test_size=0.2, random_state=42)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    opt = optimizer_
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='accuracy')

    history = model.fit(train_scaled, train_target, epochs=epochs, batch_size=128, validation_data=(val_scaled, val_target))

    return history, model

#%%231113

def make_conv2d(train_data, train_target, val_data, val_target, user_layer=None, optimizer_='adam', epoch=20):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=train_data.shape[1:]))
    model.add(keras.layers.MaxPooling2D(4))
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(7))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    if user_layer:
        model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(10, activation='softmax'))
    opt = optimizer_
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='accuracy')
    save_best_model = keras.callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    history = model.fit(train_data, train_target, epochs=epoch, validation_data=(val_data, val_target), callbacks=[save_best_model, early_stopping])
    return model, history