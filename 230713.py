# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:04:51 2023

@author: rnjsd
"""
import time # 시간측정 라이브러리

#%%
# 암호입력하기
passwrds = input('암호를 영어, 숫자, 특수기호 !, @, #, $를 포함하도록 입력하세요. :')

for i in range(1,6):
    disc_word_alpha = False
    disc_word_numeric = False
    disc_word_symbol = False    
    if len(passwrds) >= 4:
        for word in passwrds:
            if word.isalpha() == True:
                disc_word_alpha = True
            elif word.isnumeric() == True:
                disc_word_numeric = True
            elif word in '!@#$':
                disc_word_symbol = True
            elif word not in '!@#$':
                disc_word_symbol = False
        if disc_word_alpha and disc_word_numeric and disc_word_symbol == True:
            print('암호 입력에 성공하였습니다.')
            break
        elif i == 5:
            print('암호 입력 기회를 모두 사용하였습니다.')    
        elif disc_word_alpha != True:
            passwrds = input('영어를 포함시켜 주세요. :')
        elif disc_word_numeric != True:
            passwrds = input('숫자를 포함시켜 주세요. :')
        elif disc_word_symbol != True:
            passwrds = input('특수문자 !, @, #, $를 포함시켜 주세요 :')
    else:
        passwrds = input('암호는 4글자 이상 입력해주세요. :')
#%%
bool_list = [True, True, False]
print(all(bool_list)) # all() : 전부 True일 경우 True 반환
print(any(bool_list)) # any() : 리스트중 하나라도 True일 경우 True 반환
list_data = [1, 2, 3, 4, 5]
print(all(i < 5 for i in list_data))
print(any(i < 2 for i in list_data))
file_list = ['sample_data/README.md',
 'sample_data/anscombe.json',
 'sample_data/california_housing_train.csv',
 'sample_data/mnist_test.csv',
 'sample_data/mnist_train_small.csv',
 'sample_data/california_housing_test.csv']
print(file_list[1])
print(file_list[:3])
n_list = list(range(1, 10))
print(n_list)
#%%
sums = 0
for i in n_list:
    sums += i
print(sums)
#%%
def print_list(file_lists):
    for i,j in enumerate(file_lists):
        print(i, j)
print(print_list(file_list))
file_list[1:3] = ['sample_data/read', 'sample_data/output']
print(print_list(file_list))
sample_list = []
print(dir(sample_list))
sample_list.append(file_list)  #appned는 뭉텅이로 extend는 음절별로 떼옴 -> list끼리 합칠때는 extend, list에 str합칠땐 append 사용 추천
print(sample_list)
print(len(sample_list))
sample_list.clear()
sample_list.extend(file_list)
print(sample_list)
sample_list.append('python')
print(sample_list)
sample_list.extend('python')
print(sample_list)
temp = 'test python program'
sample_list.extend([temp])
print(sample_list)
#%%
sample_list = []
start1 = time.time()
sample_list.append('테스트용')
ap_time = time.time()-start1

sample_list = []
start2 = time.time()
sample_list.extend(['테스트용'])
ex_time = time.time()-start2

if ap_time > ex_time:
    print('extend가 빠름')
elif ap_time == ex_time:
    print('속도 같음')
else:
    print('append가 빠름')
#%%
sample_list = []
sample_list.extend(file_list)
sample_list.insert(2, 'python')
print(sample_list)
sample_list.remove('python')
print(sample_list)
sample_list.pop() #pop(i) i 인덱스에 해당하는 줄 삭제. 기본값은 마지막줄
last_element = sample_list.pop()
print(last_element)
sample_list.append(last_element)
print(sample_list)
sample_list.sort()
print(sample_list)
file_list_2nd = sample_list.copy() #복사 명령어 file_list_2nd = sample_list 같이 copy 명령어 사용하지 않을경우 file_list_2nd 리스트를 수정하면 sample_list 리스트도 같이 수정된다.
                                   #리스트가 [1, 2, [3,4, 11, 21, 'test', [12, 32]]] 처럼 []가 3개 이상일경우 보호 되지 않을 수 있음. 딥카피를 써야함

print(file_list_2nd)
file_list_3rd = sample_list
print(file_list_3rd)
file_list_2nd.pop()
print(file_list_2nd)
print(sample_list)
file_list_3rd.pop()
print(file_list_3rd)
print(sample_list)
sample_list.extend(file_list)
print(sample_list)
sample_list.pop(2)
print(sample_list)
dummy_set = set(sample_list) #중복된 변수 삭제 명령어
print(dummy_set)
print(type(dummy_set))
re_list = list(dummy_set)
print(re_list)
print(sample_list)
sample_list = sample_list + file_list
print(sample_list)
print(sample_list.count('sample_data/README.md'))
#%%
def print_list(file_lists):
    for i,j in enumerate(file_lists):
        print(i + 1, j)

std = ['이황', '이이', '원효']
new_std = input('전학 온 학생은 누구입니까? : ')
std.append(new_std)
std.sort()
print(print_list(std))

#%%
num_list = []
for i in range(30):
    num_list.append(i**2)
print(num_list)
#%%
a = [i**2 for i in range(30)]
print(a)
#%%
fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = []

for x in fruits:
    if "a" in x:
        newlist.append(x)

print(newlist)
#%%
odd = []                      #홀수 제곱 구하기
for i in range(30):
    if i % 2 != 0:
        odd.append(i**2)
print(odd)

odd_list = [i ** 2 for i in range(30) if i % 2 != 0]
print(odd_list)
#%%
seq1 = 'abc'
seq2 = (1, 2, 3)
[(x, y) for x in seq1 for y in seq2]
#%%
"python_module.py"