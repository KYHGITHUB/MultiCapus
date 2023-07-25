# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""
import python_module as pm
import sys
import collections
import hashlib # 암호화, 단방향
import os
#%% 230718

#%%
pm.moves(2, True)
pm.prime(17)
print(pm.prime_number(15))
print(pm.find_primes(20))
name = ["토미", "지미", "낸시", "불독"]
fes = ["OT", "CONCERT", "MT", "PLAY"]
print(pm.nameFes(name, fes))
print(pm.random_fes(name, fes))
print(pm.random_fes2(name, fes))
'''
for i in range(10):
    print(random.randint(1, 6))  #randint(i, j) -> j부분이 n-1이 아님 n임
'''
numb = [11, 15, 2, 7]
target = 9
print(pm.sums(numb, target))
print(pm.targetIndex(numb, target))
print(pm.find_indexes(numb, target))
print(pm.mysums(1,2,3,4))
print(pm.gamble(10, 100, 1))
#%%
sentence = '''God, give me grace to accept with serenity
the things that cannot be changed,
Courage to change the things
which should be changed,
and the Wisdom to distinguish
the one from the other.'''
#%%
#print(sentence)
text_path = 'test_sentence.txt'

f = open(text_path, 'w')
f.write(sentence)
f.close()
with open(text_path, 'r') as f:
    test_string = f.read()
    print(test_string)
#%%
with open(text_path, 'w') as f:
    f.write(sentence)
#%% 230719
with open('s.txt', 'r') as s:
    s_text =[line for line in s]
s_text.sort()
print(''.join(s_text))

sample_list = ['good', 'very good!', 'excellent', 'nice!']
print(sorted(sample_list)) # sorted는 sample_list에 저장하지않음
print(sorted(sample_list, key = lambda x: x[1]))   # sorted 함수에 있는 key라는 설정값울 사용해서 lambda x에 x[1]에 해당하는 값을 기준으로 sort
print(sorted(sample_list, key = len))  # 길이를 기준으로 정렬
sample_list.sort(key = lambda x: x.split()[0]) # sort는 sample_list에 저장함
print(sample_list)

with open('s.txt') as f:
    lines = f.readlines()
lines.sort(key = lambda x: x.split()[1])
print(''.join(lines))
with open('s.txt') as f:
    lines = f.read()

print(lines.splitlines()) # 줄마다 구분. 즉, \n으로 구분

with open('s.txt') as f:
    lines = f.readlines()
t_lines = []
for i in range(0, len(t_lines), 3):
    print(' '.join(t_lines[i:i+3]))

with open('s.txt') as f:
    lines = f.read().split()
for i in range(0, len(lines), 3):
    print(' '.join(lines[i:i+3]))

ip_group = {}
with open('log_webserver.txt') as f:
    for line in f:
        ip, url, times = line.split(':')
        if ip not in ip_group:
            ip_group[ip] = []
        ip_group[ip].append(url)
for i in ip_group:
    print(i)
    k = collections.Counter(ip_group[i])
    for url, count in k.items():
        print(url, '사이트에 접속한 횟수는', count, '회 입니다')
    print('=' *50)


password = 'password_my_2'
encrypted1 = hashlib.sha1(password.encode()).hexdigest()
print(encrypted1)

pm.savePasswd('권용현','abcdef')
pm.savePasswd('guest','12345')
print(pm.checkIfUservalid('guest', '12345'))
os.remove('access.text')
print(__import__('pandas'))
file_path = 'D:\\anaconda\\lib\\site-packages\\pandas\\'



#%%
