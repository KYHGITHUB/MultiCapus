# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""

import python_module as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

'''
#%%230804

path = 'D:\.spyder-py3\class file\example_data\example_data'
file_list = os.listdir(path)
print(file_list)
df = pd.read_csv('D:\.spyder-py3\class file\example_data\example_data\ex5.csv')
print(df)
df = pd.read_csv(path+'\ex5.csv', index_col = 0)
print(df)
df = pd.read_csv(path+'\ex5.csv', index_col = [0])
print(df)
print(pd.isnull(df))
print(df.info())
print(df.describe())

df.to_csv('230804테스트용.txt', index=False)
print(os.listdir())
df.to_csv('230804테스트_header=False.txt', index=False, header=False)
print(os.listdir())
df.to_csv('확인용.txt', sep ='|')
print(os.listdir())

result_yes = re.match(r'life', 'life is good')
print(result_yes)

print(re.search(r'so', 'Life is so good. so wonderful'))

number = 'My number is 511223-1****** and your is 521012-2******, 598278'
print(re.findall('\d{6}', number))

example = '이동민 교수는 다음과 같이 설명했습니다.(이동민, 2019). 그런데 다른 학자는 이 문제에 대해서 다른 견해를 가지고 있었습니다(최재영, 2019). 또 다른 견해도 있었습니다(Lion, 2018).'
print(re.findall(r'\(.+?\)', example))
sentence = 'I have a lovely dog, really. I am not telling a lie. What a pretty dog! I love this dog.'
print(re.sub(r'dog', 'cat', sentence))
words = 'I am home now. \n\n\nI am with my cat.\n\n'
print(words)
print(re.sub('\n', '', words))
'''
with open('friends101.txt') as f:
    script101 = f.read()
print(script101)
line = re.findall(r'Monica:.+', script101)
print(line)
with open('friends101_monica.txt', 'w') as f:
    f.write('\n'.join(line))
    
char = re.compile(r'[A-Z][a-z]+:')
name_script101 = re.findall(char, script101)
print(list(set(name_script101)))
with open('chracters.txt', 'w') as f:
    f.write('\n'.join(list(set(name_script101))))

character = [x[:-1] for x in list(set(re.findall(r'[A-Z][a-z]+:', script101)))]
print(character)
re.findall(r'\([A-Za-z].+?[a-z|\.]\)', script101)[:6]
sentence = '(가asknasdk.), (qwㄴㅁ어ㅁㄴ우ㅏㅁㄴ.), (qwejnaskldxzkcnasd,asdasqwkqwnkeqwelsadkasd)'
print(re.findall(r'\([A-Za-z].+?[a-z|\.]\)', sentence))

with open('friends101.txt') as f:
    sentence = f.readlines()
    
lines = []
for i in sentence:
    if re.match(r'[A-Za-z]+:', i):
        lines.append(i)
print(lines[:4])

would_list = []
for i in lines:
    if re.search('would', i):
        would_list.append(i)
print(would_list)
print(re.match('[0-9]', '567').group())     # [0-9] == \d
print(re.match('[1-4]', '45367').group())
print(re.match('5[1-4]', '51267').group())
print(re.match('[0-9]+', '512367').group())
print(re.search('는.+[\d]', '나는 낭만 고1양이').group())
print(re.search('는.+[\d]', '나는 낭만 고1양이').span())
print(re.findall(r'[a-z]+', 'python 3 version program'))
p = re.compile(r'([A-Za-z]\w*)\s*=\s*(\d+)')
print(p.search('a = 123').group())
re.match(r'a.*b', 'acbd, asefsdgweryerb').group()
re.match(r'\s\w*', ' abcdefg').group()



