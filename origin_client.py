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
#%% 230713

pm.test()
result = pm.adds([11, 32, 67, 89])

print(result)
result

sample_list = ['sample_data/README.md',
 'sample_data/read',
 'sample_data/output',
 'sample_data/mnist_test.csv',
 'sample_data/mnist_train_small.csv',
 'sample_data/california_housing_test.csv']
pm.print_list(sample_list)
print(sample_list)
#%%
print(pm.adds1(17))
#%%
print(pm.create_dict(5))
print(pm.create2_dict(5))
pm.colab_contents_230713()

#%%
#test_string = '''Humans are odd. They think order and chaos are somehow opposites and try to control what won't be. But there is grace in their failings.'''
print(pm.change_string(test_string))
print(pm.splitString(test_string))
#%% 230714
pm.func_1(1, 2)
print(pm.create_dict(5))
pm.func_1(3,5)
print(pm.add_func(4, 3))
a = 11
b = 23
pm.no_return_func(a, b)
pm.in_out_gugudan(19, 28)
pm.in_out_gugudan2(11,13)
pm.in_out_gugudan3(5, 7)
print(pm.nothing())
print(pm.nothing1())
type(pm.nothing1())
print(pm.nothing2())
print(pm.minuss(170, 60))    #positional argument
print(pm.minuss(60, 170))
print(pm.minuss(height=170, width=60))  #keyword argument)
print(pm.minuss(width=60, height=170))
print(pm.minuss(170, width=60))    ##일반적으로 함수를 호출할 때 키워드 인수의 위치는 위치 인수 이후이다.
pm.varg(2, 3)
pm.varg(2,3,4,5,6,7,8)
pm.printf("I've spent %d days and %d night to do this", 6, 5)
i=3
j=4
print('%d * %d = %d' % (i, j, i*j))
pm.printf('내 나이는 %d세 이고 태어난 연도는 %d년 이다.', 32,1992)
sample_list = [10, 20, 30]
pm.len_user(sample_list)
pm.f(width = 10, height = 5, depth = 10, dimension = 3)
pm.g(1,2,5,6,7,c=8,d=9)
pm.g(10,20,[1,2,3,4], {11:22, 22:33}, c=2, k=3)
q = 10
w = (1,2,3)
e = [1,3]
r = {'df':123, 'iu':3256}
pm.g(a,w,e,r)
pm.g(a,w,e,*r)
pm.g(a,w,e,**r)
pm.h(1, 2, 3)
a =  (10, 20, 30)
pm.h(a[0], a[1], a[2])
pm.h(*a)           #tuple -> * (하나)
dargs = {'a':1, 'b':2, 'c':3}
pm.h(**dargs)      #dict -> ** (두개)
args = (10, 20)
dargs = {'c':30}
pm.h(*args, **dargs)
pm.h(*{'a':1, 'b':124,'c':12515})  # dict 받아들일때 *이면 kes값, **이면 valus값 받아들임
b = [8,9,10]
print(b)
print(pm.ff(b))
print(b)
print(pm.n_squared([2,3,4],2))
target_list = [1, [11, 21], 3, 4]
pm.edit_list(target_list)
target_list
target_list = [1, [11, 21], 3, 4]
pm.gen_edit(target_list)
target_list = [1, [11, 21], 3, 4]
pm.copy_edit(target_list)
print(target_list)
target_list = [1, [11, 21], 3, 4]
pm.copy_edit_2nd(target_list)
print(target_list)
target_list = [1, [11, 21], 3, 4]
pm.deep_copy_edit(target_list)
print(target_list)
target_list = list(range(6))
pm.no_change_edit(target_list)
target_list
sample_list = list(range(10000000))
pm.time_check(sample_list)
pm.lotto(30)

#%% 230717
target = 100
result = pm.create_list(target)
print(result)

target = 'python'
pm.app_ext_list(sample_list, target)
#%% 230717
target = 100
result = pm.create_list(target)
print(result)

sample_list = result
target = 'python'
pm.app_ext_list(sample_list, target)
print(sample_list)

items_list = [12, 567, 34, 9, 17]
print(pm.min_list(items_list))
print(pm.max_list(items_list))

result = pm.create_dict(9)
print(result)

keys_list = ['name', 'year', 'attr']
values_list = ['joker', 2019, 'villlain']
dict_result = pm.add_dict(result, keys_list, values_list)
print(dict_result)
dict_result2 = pm.add2_dict(result,keys_list, values_list)
print(dict_result2)

cannon_result = pm.cannon_add_dict(dict_result, (10, 9), 1729)
print(cannon_result)

key_list = ['one', 'two']
value_list = [100, 200]
roc_2 = pm.cannon_add_dict(cannon_result, key_list, value_list)
print(roc_2)
pm.outer()
c1 = pm.makeCounter()
print(c1())
print(c1())
print(c1())
print(c1())
c2 = pm.makeCounter()
print(c2())
c3 = pm.makeCounter()
print(c3())
print(c3())
print(c3())
f1 = pm.quadratic(1, 2, 1)
print(f1(1))
f2 = pm.quadratic(1, 6, 9)
print(f2(1))
print(pm.divisor(100))
print(pm.add_(10)) # 10 + add_(9), 10 + 9 + add_(8)
print(pm.add_recur(5))
print(pm.gcd(78696, 19332))
pm.hanoi_tower(3, 'left', 'right', 'middle')
#%% 230718
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
