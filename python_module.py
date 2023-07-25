# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import numpy as np
import random
import hashlib
import os
#from google.colab import output #코랩에서 작동
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