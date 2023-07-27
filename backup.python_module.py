# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import copy
import time
import random
import hashlib
import os
from decimal import Decimal
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