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
import mymath
#from mymath import area, mypi
#from mymath import *
from string import punctuation as punc
import matplotlib.pyplot as plt
import requests # 웹서버 가지고 오는 라이브러리
import sys_path_add
from module_test import module_test as mt
from  datetime import datetime, timedelta
import numpy as np
import pandas as pd
import platform
from matplotlib import font_manager, rc
import seaborn as sns
import pandas_datareader.data as web
#   %matplotlib -> 로컬에선 쓰지않고 코랩에선 라인에 그래프가 포함되게 해달라는
#                  명령어.
import copy
import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
pd.set_option('display.max_columns', None)      # 데이터프레임 끝까지 보여주기
pd.set_option('display.max_rows', None)
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
test_string = '''Humans are odd. They think order and chaos are somehow opposites and try to control what won't be. But there is grace in their failings.'''
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
#%%230727
prices = [7, 1, 5, 3, 6, 4]
plt.plot(prices)        #conda install matplotlib 프롬프트로 설치
plt.show()

with open('data.txt', 'w') as f:
    f.write('1 2 3\n4 5 6\n7 8 9')

print(pm.change_list('data.txt'))
print(pm.change2_list('data.txt'))
print(pm.trans_matrix('data.txt'))
#%%
prices = [7, 1, 5, 3, 6, 4]
print(pm.price_profit(prices))

data = {
'Eoleumgol' : [35.570678, 128.986135],
'Bangujeong' : [37.867614, 126.752505], 'Abgujeong':[37.531713, 127.029154],
'Guemgangsonamusup': [36.985459, 129.205468],
'Heonanseolheonmyo': [37.425569, 127.296012],
'Baegdamsa': [38.164890, 128.374023],
'Moagsan mileugbul' : [35.723051, 127.053816],
'Hailli' : [37.680120, 126.398192],
'Ieodo': [32.116883, 125.166683],
'Bughansan' : [37.659318, 126.9775415],
'Ondalsanseong' : [37.057707, 128.484972],
'Cheonglyeongpo' : [37.176118, 128.445583],
'Hansanseom' : [34.816761, 128.423040],
'Haeinsa' : [35.801479, 128.098052],
'Sancheonjae' : [35.275175, 127.849891],
'Seomjingang' : [34.963452, 127.760620],
'Baegheungam' : [35.994240, 128.778653],
'Guksaseonangdang': [37.696354, 128.753741],
'Mudeungsan' : [35.134134, 126.988756],
'Busanseong' : [36.268112, 126.914802],
'Cheolsanri' : [37.808038, 126.450912], 'Odusan' : [37.773131, 126.677203]
}

pm.MarkerMap(data).save('data.html')

my_dict = {'Home':[36.11823463264378,128.3507192744579],
           'visitPlace':[36.108057626856926, 128.41995192033562 ],
           'favoritePlace':[35.15845558589585, 129.15990774436943 ]}

pm.MarkerMap(my_dict).save('myplace.html')

print(pm.add(5))
print(pm.add(3))
print(pm.add(8))
print(pm.add(10))
cal1 = pm.Calculator()
print(cal1.add(5))
print(cal1.add(7))
print(cal1.add(108))

url = 'http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json?key=c3a0fa760324363ef7cde2afa0d73297&targetDt=20230726'
resp = requests.get(url)
print(resp)
data = resp.json()
print(data)
print(data.keys())
print(data['boxOfficeResult'])
print(type(data['boxOfficeResult']))
print(data['boxOfficeResult'].keys())
print(data['boxOfficeResult']['dailyBoxOfficeList'])
pm.index_data(data)
mt.index_data(data)
#%%230728
pm.MarkerMap('경기도공중화장실현황(제공표준).json').save('경기도화장실.html')
pm.MarkerMap2('경기도으뜸맛집현황.json').save('경기도맛집.html')

a = pm.MyClass()
a + 3

mc1 = pm.MyClass2('권용현', 32)
print(mc1.name)
print(mc1.age)

mc2 = pm.MyClass3()
print(mc2.age)
print(mc2.name)

mc3 = pm.MyClass4()
print(mc3.age)
mc4 = pm.MyClass4(192,'김창창')
print(mc4.name)
print(mc4.age)

mc5 = pm.MyClass5('010900')
print(mc5.phone)
print(mc5.age)
print(mc5.print_attr(1292))

b = pm.B()
print(pm.B.__bases__)
b.f()
print(dir(pm.B))
print(dir(b))

mc6 = pm.MyClass6()
mc6.set(100)
mc6.tem()
pm.MyClass6.set(mc6, 200)
print(pm.MyClass6.get(mc6))

life_io = pm.Life()
del(life_io) 

v1 = pm.Var()
v2 = pm.Var()
print(v1.c_mem, v2.c_mem)
v1.c_mem = 50
print(v1.c_mem, v2.c_mem)

fc1 = pm.FourCal(10, 100)
print(fc1.add(), fc1.mul(), fc1.sub(), fc1.div())

fc2 = pm.safeFourCal(100, 0)
print(fc2.div())
fc3 = pm.safeFourCal(100, 10)
print(fc3.div())
#%%230731

ms1 = pm.MyStr('a, b, c')
print(ms1)
print(ms1 + ' :d')
print('z: ' + ms1)
print('z: ' + ms1 + ' :d')

fc1 = pm.FourCal2(10, 50)
print(fc1 + 5)
print(fc1 - 5)
print(fc1 * 5)
print(fc1 / 5)
print(5 * fc1)

c1 = pm.A()
print(c1)

c2 = pm.B2()
print(c2)

c3 = pm.C()
print(c3)

sr1 = pm.StringRepr()
sr1
print(sr1)
print(str(sr1))
print(repr(sr1))

sr2 = pm.StringRepr2()
print(sr2)
print(str(sr2))
print(repr(sr2))

sr3 = pm.StringRepr3()
print(sr3)
print(str(sr3))
print(repr(sr3))

eval('10 + 20') # eval() 함수에 의하여 같은 객체로 재생성 될 수 있는 문자열 표현
a = '10 + 20'
print(eval(a))
print('abc')
b = '''print('abc')'''
eval(b)

sr4 = pm.StringRepr4()
print(sr4.i)
q = eval(repr(sr4))
print(q.i)
print(eval('pm.StringRepr4(23)').i)
obj = pm.MyClass7(10, 20)
print(obj)
print(repr(obj))

acc1 = pm.Accumulator()
print(acc1(1,2,3,4,5))

a = pm.A()
b = pm.B()
pm.check(a)
pm.check(b)
callable(b)

fact = pm.Factorial()
print(fact(5))
for i in range(10):
    print(f'{i}! = {fact(i)}')

a = pm.MyStr2("I like python and python")
print(a)
print(a + " stuff")
print(a - "python")

a = pm.Test_no_getitem()
print(a._numbers)
print(a[3])

my_list = pm.MyList(['red', 'blue', 'green', 'black'])
print(my_list[0])
print(my_list[2])
print(my_list['red'])
print(my_list['green'])

s = pm.Square(10)
print(s[3])
print(s[9])

p1 = pm.Person('홍길동', 1498)
print(p1)
e1 = pm.Employee('마석도', 5564, '형사', 500)
print(e1)
print(e1.salary)
print(e1.position)

print(dir(p1))
l1 = [x for x in dir(p1) if not x.startswith('__')]
print(l1)

print(dir(e1))
l2 = [x for x in dir(e1) if not x.startswith('__')]
print(l2)

e2 = pm.Employee2('마석도', 5564, '형사', 500)
print(e2)

p1 = pm.Point(3, 5)
print(p1)
c1 = pm.Circle(3, 4, 5)
print(c1)
c2 = pm.Cylinder(3, 4, 5, 6)
print(c2)
print(c2.area())
print(c2.volum())
e4 = pm.Employee3('마석도', 5564, '형사', 299)
print(e4)
pm.Employee3.mro()
#%%230801
b1 = pm.B(10)
b1.print_x()
print([x for x in dir(b1) if not x.startswith('__')])
b2 = pm.B(10)
b2.setdata(77)
b2.print_y()
print([x for x in dir(b2) if not x.startswith('__')])

e1 = pm.Employee('마석도', 5584)
e1.print_info()
print([x for x in dir(e1) if not x.startswith('__')])
e1.setdata('광수대', 200)
print([x for x in dir(e1) if not x.startswith('__')])

s1 = pm.Stack()
s1.push(4)
print(s1)
s1.push('안녕')
print(s1)

q1 = pm.Queue()
q1.enqueue(1)
q1.enqueue(10)
print(q1)
print(q1.dequeue())
print(q1)

md1 = pm. MyDict()
print(md1.keys())
md2 = pm.MyDict({10:100, 21:200, 3:300})
print(md2.keys())

for each in (pm.Dog(), pm.Duck(), pm.Fish()):
    each.cry()

L = pm.OrderedList([3, 10, 2])
print(L)
L.append(5)
print(L)
L.extend((4, 8, 20))
print(L)

c = pm.Counter()
print(c.incr())
print(c())
c2 = pm.Counter()
c_b = pm.Counter.incr
print(c_b(c2))

book1 = pm.Book()
book1.set_info('하얼빈', '김훈')
book1.print_info()
book1.set_info('채식주의자', '한강')
book1.print_info()
data_list = [('하얼빈', '김훈'), ('채식주의자', '한강')]
for each in data_list:
    book1 = pm.Book()
    book1.set_info(each[0], each[1])
    book1.print_info()

obj_song = pm.Song()
obj_song.set_song('취중진담', '발라드')
obj_song.print_song()
obj_singer = pm.Singer()
obj_singer.set_singer('김동률')
obj_singer.hit_song(obj_song)
obj_singer.print_singer()

p1 = pm.Personal('홍길동', '1443년', '?')
p1.append("전화하지마세요")
print(p1)
p1.present()
print(p1.name)
p1_dict = {p1.name : p1}
print(p1_dict)
p1_dict['홍길동'].present()
p1_dict['홍길동'].append('소설속 인물')
print(p1_dict)
p1_dict['홍길동'].present()

Rossum = pm.Personal("Guido van Rossum", 1991, '모르는 사람')
Rossum.append('파이썬 발명가')
print(Rossum)

result = pm.wrapper1(pm.myfunc1)
result()
result = pm.myfunc2

print(pm.say())

print(pm.add1(1, 2))

print(pm.add2(10, 20))

print(pm.myfunc3(10, 20))
print(pm.myfunc4(10,20))

#%%230802
#%% datetime
dt = datetime(2023, 8, 2, 13, 5, 20)
print(dt)
print(dt.year)
print(dt.month)
print(dt.date())
print(dt.time())
print(dt.date(), dt.time())

print(dt.strftime('%Y%m%d %H:%M'))
datetime.strptime('20230802', '%Y%m%d')
dt
print(dt.replace(minute=5, second=47))
dt = datetime.now()
dt.microsecond
delta = datetime.now() - dt
print(delta)
print(type(delta))
print(dt + timedelta(hours=2))
print(dt + timedelta(30))

print(dir(timedelta))
print(f'현재시간 - 과거시간 = {delta.seconds}')
base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
print(f'base_time : {base_time}')
future_time = datetime(2023, 12, 26)
print(f'future_time : {future_time}')
print(f'(future_time - base_time).total_seconds() = {(future_time - base_time).total_seconds()}')
diff = future_time - base_time
print(diff)
print(diff.total_seconds() / 3600)
for i in range(diff.days):
    print(i)

#%%     
x_seq = list(range(10))
data = np.arange(10)
print(x_seq)
print(data)
for i in x_seq:
    print(i*3)
y = data ** 2
x = data
plt.plot(data, data**2)
plt.show()

fig = plt.figure()
axes = fig.add_subplot()
axes.plot(x, y)
plt.show()

fig, axes = plt.subplots()
axes.plot(x, y)
plt.show()

x = np.arange(-10, 11, 1)
x
fig, ax = plt.subplots(figsize = (4, 4))
plt.plot(x, x*x)
plt.plot(x, 2*x)

t = np.arange(0, 5, 0.5)
plt.figure(figsize=(10,6))
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'gs')
plt.plot(t, t**3, 'b>')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(np.random.randn(50).cumsum(), 'k--')       # randn : 정규분포에서 추출된 난수들, cumsum : 누적 합계
_ = ax1.hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
ax2.scatter(np.arange(30), np.arange(30)+ 3 * np.random.randn(30))

fig, axes = plt.subplots(2, 2, sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.randn(500), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace=0, hspace=0)

x = np.arange(10)
y = x*10 + 2
fig, ax = plt.subplots()
ax.plot(x, y, 'g--')

fig, ax = plt.subplots()
ax.plot(x, y, linestyle = '--', color = 'g')
plt.plot(np.random.randn(30).cumsum(), 'ko--')
plt.plot(np.random.randn(30).cumsum(), color = 'k', linestyle = 'dashed', marker = 'o')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30)
ax.set_title('My first plot', fontsize = 18)
ax.set_xlabel('Stages', fontsize = 13)

props = {'title' : 'My first plot', 'xlabel' : 'Stages'}
ax.set(**props)
#%%
x = np.arange(1, 10, 0.1)
loss = lambda x : np.exp(-x)
acc = lambda x : -np.exp(-x)

x1 = np.random.randn(len(x))

fig, loss_ax = plt.subplots(figsize=(8,6))

acc_ax = loss_ax.twinx() # x 축을 공유하는 새로운 axes 객체를 만들어 준다. 결과적으로 x축은 같고 y측만 다른 그래프가 생긴다.

loss_ax.plot(x, loss(x), 'y', label = 'train loss')
loss_ax.plot(x, loss(x-x1/5), 'r', label='validation loss')

acc_ax.plot(x, acc(x), 'b', label='tarina acc')
acc_ax.plot(x, acc(x-x1/7), 'g', label='val acc')
for label in acc_ax.get_yticklabels(): # y 축 tick 색깔 지정
    label.set_color("blue")

loss_ax.set_xlabel('epoch', size=20)
loss_ax.set_ylabel('loss', size=20)
acc_ax.set_ylabel('accuray', color='blue', size=20)

loss_ax.legend(loc='upper right')
acc_ax.legend(loc='lower right')
fig.savefig('test1.png')
plt.show()
#%%
x = np.arange(0,10, np.pi/100)
f = lambda x : np.sin(x)+x/10

XTN=[r'$0 \pi$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$',r'$5\pi/2$',r'$3\pi$']
plt.figure(figsize=(8,6))
plt.plot(x, f(x))
plt.title('Plot Exercise',position=(0.5,1.04), fontsize=20)
plt.xlabel (r'$x$',fontsize=20)
plt.ylabel (r'$f (x) = sin(x) + x$', fontsize=20)
plt.xticks(np.arange(0.0,10.0,np.pi/2), labels=XTN, fontsize=15)
plt.yticks(np.arange(-1,2.1,np.pi/5), fontsize=15)
plt.text(0.8,-0.3,'Test Messages', color='k', fontsize=18)
plt.grid()  # 배경에 줄 쳐주는것
plt.tight_layout()  # title이 있으면 그래프 외곽선에 맞추기
plt.savefig('test2.png')
plt.show()


x = [10, 20 ,30]
array_x = np.array(x)
print(array_x)
obj = pd.Series(array_x)
print(obj)
print(type(obj))
print(len(dir(obj)))
print(obj.values)
print(type(obj.values))
print(obj.index)
for i in obj.index:
    print(i, obj[i])
obj2 = pd.Series([4, 7, -5, 3], index = ['a', 'b', 'c', 'd'])
print(obj2)
print(obj2.index)
for i in obj2.index:
    print(i, obj2[i])
print(obj2['a'])
obj2['d'] = 6
print(obj2)
print(obj2[obj2>0]) # 외우기
data = pd.Series([-1, 0 ,1], index=['a','b','c'])
print(data[data<0])
print(obj2**2)
print(data*2)
print(np.exp(data))
print('c' in data)
print(1 in data)
print(data[0])
sdata = {'Ohio':35000, 'Texas':71000, 'Oregon':1600, 'Utah':5000}
print(type(sdata))
obj3 = pd.Series(sdata)
print(obj3)
print(obj3['Ohio'])
print(list(obj3.index))
std__ = ['California', 'Ohio', 'Oregon', 'Texas']
sdata
obj4 = pd.Series(sdata, index = std__)
print(obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj4.isnull())
print(obj4.notnull())
obj3
obj4
print(obj3 + obj4) # NaN == None
obj4.name = 'population'
print(obj4)
obj4.index.name = 'state'
obj
obj.index = ['Bob', 'Stetve', 'Jeff']
print(obj)

data = {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year':[2000, 2001, 2002, 2001, 2002, 2003],
        'pop':[1.5, 1.7, 3.6, 2.4, 2.9, 3.2]
        }
df = pd.DataFrame(data)
print(df)
인구1 = {'인구':{'하나':1542, '여섯':9776, '둘':1535},
         '지역':{'하나':'대전', '여섯':'서울', '둘':'대전', '셋':'대전'}}
print(인구1)
print(인구1.keys())
print(인구1['인구'].keys())
print(pd.DataFrame(data, columns=['year', 'state', 'seoul', 'pop']))
df2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'],
                   index = ['one', 'two', 'three', 'four', 'five', 'six'])
print(df2)
print(df2.index)
print(df2.columns)
print(type(df2))
print(type(df2['state']))
print(df2['state'])
print(df2.state)   # == df2['state']
print(df2['pop'])
df5 = pd.DataFrame(인구1)
print(df5)
print(df5.인구)
print(df5['인구'])
print(df5.loc['여섯'])
print(df5.iloc[1])
df2['debt'] = 16.5
print(df2)
copy_df2 = copy.copy(df2)
copy_df2['debt'] = ['틀렸어', 'a', 1, 'spiderman', 'superman', 7]
print(copy_df2)
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
print(val)
df2['debt'] = val
print(df2)
df2['stern'] = df2['state'] == 'Ohio'
print(df2)
del df2['stern']
print(df2)
print(data)
pops = {'Nevada':{2001:2.4, 2002:2.9}, 'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}
df3 = pd.DataFrame(pops)
print(df3)
print(df3.T)
print(pd.DataFrame(pops, index=[2001, 2002, 2003]))
print(df3['Ohio'][:-1])
pdata = {'Ohio':df3['Ohio'][:-1],
         'Nevada':df3['Nevada'][:2]}
print(pdata)
print(pd.DataFrame(pdata))
print(df5)
df5.index.name = '번호'
df5.columns.name = '분류'
print(df5)

data = range(3)
s1 = pd.Series(data, index = ['a', 'b', 'c'])
s2 = pd.Series(data, index = ['a', 'b', 'c'])
s3 = pd.Series(data, index = ['a', 'b', 'c'])
print(s1)
print(s2)
print(s3)
print(s1.rename(index = {'a':'A'}))
s2.index = '구분'.join(s2.index).replace('a', 'A').split('구분')
s3.index.values[0] = 'A'
print(s2)
print(s3)
df5
print('인구' in df5.columns)
print('하나' in df5.index)
print(df5.loc['하나'])
df = pd.Series([3, -8, 2, 0], index=['d', 'b', 'a', 'c'])
print(df)
df.reindex(['a', 'b', 'c', 'd', 'e'])
print(np.arange(6).reshape(3,2))   #   reshape(row, columns)
df = pd.DataFrame(np.arange(6).reshape(3,2), index = range(0, 5, 2), columns = ['A', 'B'])
print(df)

#%%230803



df = pd.DataFrame(np.arange(6).reshape(3,2),
                  index = range(0,3), columns = ('A', 'B'))
print(df)
print(df.rename(index = {0 : 'a'}))
df.rename(index = {0 : 'T'}, inplace = True)
print(df)
for i in range(3):
    print(f'{df.index[i]}의 타입은 {type(df.index[i])} 이다')
df.rename(columns = {'A':'열번호 1', 'B':'열번호 2'}, inplace =True)
print(df)

font_path = '\\.spyder-py3\\class file\\NanumGothic.ttf'
fontprop = font_manager.FontProperties(fname = font_path, size =10)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000]) #   x축 눈금
labels = ax.set_xticklabels(['하나', '둘', '셋', '넷', '다섯'],        # x축 값들 이름 정하기
                             rotation = 30)     # rotaion : 글자의 기울기
plt.xticks(fontproperties = fontprop)       # fontproperties
plt.show()

property_candidate = {'이재명':3217161, '윤석열':7745343, '심상정':1406297,
                      '안철수':197985542, '오준호':264067, '허경영':26401367, '이백윤':171800,
                      '옥은호':337062, '김동연':4053544, '김경재':2202623, '조원진':2058661,
                      '김재연':51807, '이경희':149907313, '김민찬':421648}
x = list(property_candidate.keys())
y = np.array(list(map(lambda x:x/1000, property_candidate.values())))
print(x)
print(y)
plt.figure(figsize = (8,8))
sns.barplot(x=x, y=y)
sns.set_theme(style='white', context='talk')
plt.xticks(fontproperties = fontprop, rotation = 90)
plt.show()

sns.set_theme(style="white", context="talk")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize=(10,8))
fig.subplots_adjust(hspace=0.2)  # adjust space between axes

# plot the same data on both axes
ax1.bar(property_candidate.keys(), list(map(lambda x : x/1000, property_candidate.values())), color = ['yellow', 'cyan', 'pink', 'purple'], alpha = 0.5,
edgecolor = 'black', linewidth = 2.5)
ax2.bar(property_candidate.keys(), list(map(lambda x : x/1000, property_candidate.values())),color = ['yellow', 'cyan', 'pink', 'purple'], alpha = 0.5,
edgecolor = 'black', linewidth = 2.5)

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(120000, 200000)  # outliers only
ax2.set_ylim(0, 30000)  # most of the data



# hide the spines between ax and ax2

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax1.set_title('20 대 대선 후보자 재산 현황 (단위 : 백만원)', fontproperties = fontprop, pad=20)
#ax1.set_ylabel('단위 : 백만원',labelpad=20, fontproperties = fontprop)
ax2.xaxis.tick_bottom()
ax2.set_xticklabels(property_candidate.keys(),fontproperties = fontprop,rotation = 270)

#Y축 양쪽에 빗금 넣기
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.show()

plt.rcParams['axes.unicode_minus'] = False 

if platform.system() == 'Darwin':           # paltform -> 윈도우같은걸 말함
    rc('font', family='AppleGothic')        # 초기화 값 읽어주는걸 담당
elif platform.system() == 'Windows':
    path = 'c:\\Windows\\Fonts\\malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    
    font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'       # 로컬에서는 경로 변경이 필요함
    fontprop = font_manager.FontProperties(fname=font_path, size=10)
    font_name = fontprop.get_name()         # get_name : 이름 바꾸기
    rc('font', family=font_name)
print(font_path)

df = pd.Series([3, -8, 2, 0], index=['d', 'b', 'a', 'c'])
print(df)
print(df.reindex(['a', 'b', 'c', 'd', 'e']))
df.reindex(range(4))
df1 = pd.Series(['blue', 'red', 'green'], index =[0, 2, 4])
print(df1)
df1.reindex(range(6))
print(df1.reindex(range(6), method = 'ffill')) #   ffill : f + fill 로서 앞에껄로 채운다는 의미
df = pd.DataFrame(np.arange(6).reshape(3,2),index = range(0, 5, 2), columns = ('A', 'B'))
df_ext = df.reindex(range(5), columns = ['B', 'C', 'A'])
print(df_ext)
df_ext.drop(1, inplace = True)  # indeex 지우기
print(df_ext)
print(df_ext.drop(0))
print(df_ext.drop('C', axis = 'columns'))  # axis : 축, axis = 'columns' : 축이 columns축이다 
print(df_ext.drop('C', axis = 1))
df_ext.drop(3, inplace=True)
print(df_ext)
df = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
print(df)
print(df[1])
print(df['b'])
print(df[:3])
print(df['a':'c'])     # 문자로 슬라이싱 할때는 숫자와는 달리 문자 -1 이 아닌 문자까지 간다. ex) x[5] -> 인덱스 4번까지, x['e'] -> 인덱스값이 'e'인곳 까지
df['c':'d'] = 0
print(df)
df = pd.DataFrame(np.arange(3*4).reshape(3, 4), index = ['A', 'B', 'C'], columns = ['aa', 'bb', 'cc', 'dd'])
print(df)
print(df['aa'])
print(df[['aa', 'cc', 'dd']])
print(df[:2])
print(df[df['aa']<=4])
print(df.loc[['A','B'], ['aa', 'cc']])     #   loc[행, 열]
print(df)
print(df.loc['A'])
print(df.iloc[:2, [0, 2]])

url = 'https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv'
df = pd.read_csv(url, sep='\t')
print(df)
df.info()
print(df.shape)
print(list(df.columns))
for i in df.columns:
    print(i)

path = 'D:\\.spyder-py3\\class file\\example_data\\example_data\\'
df = pd.read_csv(path + 'ex1.csv', sep = ',')
print(df)
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

#%%230807
with open('test.html', encoding = 'utf=8') as f:
    line = f.read()
line
soup = bs(line, 'html.parser')
print(soup)
print(soup.prettify())
print(soup.children)
soup_children_list = list(soup.children)
list(soup.children)[1]
print(soup_children_list)
print(soup.body)
print(soup.head)
print(soup.find_all('p'))
for i in soup.find_all('p'):
    print(i.text)

news = ' https://news.daum.net'
soup = bs(urlopen(news), 'html.parser')
print(soup)
print(soup.find_all('div', {'class':'item_issue'}))    # div 태그에서 class(key)값이 item_issue(value)인것을 찾아라
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.text)
print(soup.find_all('a')[:5])
for i in soup.find_all('a'):
    print(i.get('href'))
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.find_all('a')[0].get('href'))
    
article1 = 'https://go.seoul.co.kr/news/newsView.php?id=20200427004004&wlog_tag3=daum'
soup = bs(urlopen(article1).read(), 'html.parser')
print(soup)
for i in soup.find_all('p'):
    print(i.text)
    
news = ' https://news.daum.net'
soup = bs(urlopen(news), 'html.parser')
headline = soup.find_all('div', {'class':'item_issue'})
for i in headline:
    print(i)
for i in headline:
    print(i.text)
    soup3 = bs(urlopen(i.find_all('a')[0].get('href')).read(), 'html.parser')
    for j in soup3.find_all('p'):
        print(j.text)
for i in soup.find_all('div', {'class':'item_issue'}):
    print(i.find_all('a')[0].get('href'))
with open('link.txt', 'w') as f:
    for i in soup.find_all('div', {'class':'item_issue'}):
        f.write(i.find_all('a')[0].get('href')+'\n')

article1 = 'https://v.daum.net/v/20230807102700905'
soup = bs(urlopen(article1).read(), 'html.parser')
with open('article1.txt', 'w') as f:
    for i in soup.find_all('p'):        # p태그는 기사내용을 찾기위함이다
        f.write(i.text+'\n')

url = 'https://news.daum.net/'
soup = bs(urlopen(url).read(), 'html.parser')
headline = soup.find_all('div', {'class':'item_issue'})
headline

for i in headline:
    print(i.text)
for i in headline:
    print(i.text)
    print(i.find_all('a')[0].get('href'))
    new_url = i.find_all('a')[0].get('href')
    soup2 = bs(urlopen(new_url).read(), 'html.parser')
    for j in soup2.find_all('p'):
        print(j.text+'\n')

with open('article_total.txt', 'w',encoding = 'utf-8') as f:
    for i in headline:
        f.write(i.text)
        f.write(i.find_all('a')[0].get('href')+'\n')
        new_url = i.find_all('a')[0].get('href')
        soup2 = bs(urlopen(new_url).read(), 'html.parser')
        for j in soup2.find_all('p'):
            f.write(j.text+'\n')

url = 'https://www.chicagomag.com/chicago-magazine/january-2023/our-30-favorite-things-to-eat-right-now/'
hdr = {'User-Agent':'Mozilla/5.0'}
req = Request(url, headers=hdr)
page = urlopen(req)
soup = bs(page, 'html.parser')
soup
temp = soup.find_all('div', {'class':'article-body'})[0]
temp
f_list = []
r_list = []
p_list = []
a_list = []
temp.find_all('h2')[0].text
for f in temp.find_all('h2'):
    f_list.append(f.text)
f_list
temp.find_all('h3')[22].text.split('at')[1]
for r in temp.find_all('h3'):
    r_list.append(r.text.split('at')[1].strip())
r_list
temp.find_all('p')[0].text.index('$')
temp.find_all('p')[0].text[314]
temp.find_all('p')[0].text[314:].split()[0].strip('.')
' '.join(temp.find_all('p')[0].text[314:].split()[1:]).strip()
for p in temp.find_all('p'):
    p_index = p.text.index('$')
    p_list.append(p.text[p_index:].split()[0].strip('.'))
    a_list.append(' '.join(p.text[p_index:].split()[1:]).strip())
p_list
a_list
data = {'Food':f_list, 'Restaurant':r_list, 'Price':p_list, 'Address':a_list}
data
df = pd.DataFrame(data)
print(df)
#%%230808
df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']], index = ['준서', '예은'], columns = ['나이', '성별', '학교'])
df
print(df.index)
print(df.columns)
df.index = ['학생1', '학생2']
df.columns = ['연령', '남녀', '소속']
print(df)
df.rename(columns = {'남녀':'sex'}, inplace = True)
print(df)
df2 = df.copy()
df2.drop(index = '학생1', inplace = True)
print(df2)
print(df)
df.loc['학생1', 'sex'] = '여'
print(df)
df['소속']
exam_data = {'이름':['서준', '우현', '인아'],
             '수학':[90, 80, 70],
             '영어':[98, 89, 95],
             '음악':[85, 95, 100],
             '체육':[100, 90, 90]}
df = pd.DataFrame(exam_data)
df
df.sort_values(by = '음악')
df
df2 = df.sort_values(by = '체육')
df2
df2.index = ['a', 'b', 'c']
df2
df2.loc['a']
df2
print(df2.reset_index())
df2
print(df2.loc['a', ['음악','체육']])
print(df2.loc['a', '음악':'체육'])
df2.set_index('이름', inplace = True)
df2
print(df2.loc['인아':'서준', '수학':'음악'])
df
df.set_index('이름', inplace = True)
df
df3 = df.loc[:, '수학':'영어']
df3.reset_index(inplace = True)
print(df3)
#%%230823
x = np.arange(-5, 5, 0.1)
y =list(x)
pm.Plotbool(x)
pm.Plotsigmoid(x)
y = list(range(10))
pm.Plotbool(y)
pm.Plotsigmoid(y)


pm.Plotsigbool(x)

np.linspace(0, 3, 1000)

pm.straightPlot(x)
pm.straightPlot(y)
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = list(range(10))
pm.straightPlot(x)
pm.straightPlot(y)

x

type(np.array(y))
pm.aryAsbool(y)


file_path = 'D:\\.spyder-py3\\class file\\'
file_name = 'movies.dat'
df = pm.getDat(file_path + file_name)
name = ['movie_id', 'title', 'genres']
df.columns = name
print(df)
new_df = pm.getDummies(df)
print(new_df)

#%%230824
sr = pd.Series(range(5))
pm.catCodes(sr)
if isinstance(df_di, pd.DataFrame):
    print('시리즈입니다')
iris = sns.load_dataset('iris')
iris
fig = plt.figure(figsize=(6, 4))
df
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
print(pm.catCodes(fruits))
di = {'alphabet':['a','b','c','d','a','b',
                 'b','b','e','t'],
      'number':[1,2,3,1,1,1,5,7,3,3]}
pm.catCodes(di)
df_di = pd.DataFrame(di)
type(df_di)
df_di['alphabet'].astype('category').cat.codes
x=[{'city':'seoul', 'temp':10.0}, {'city':'Dubai', 'temp':33.5}, {'city':'LA', 'temp':20.0}]
x, x_rare= pm.dictAsvec(x)
print(x_rare)
D = [{'foo':1, 'bar':2}, {'foo':5, 'baz':4}]
pm.dictAsvec(D)
d = pd.DataFrame()
text=['떴다 떴다 비행기 날아라 날아라',
      '높이 높이 날아라 우리 비행기',
      '내가 만든 비행기 날아라 날아라',
      '멀리 멀리 날아라 우리 비행기']
if isinstance(df_di, pd.DataFrame):
    print('시리즈입니다')
pm.textAsvec(text)
text1 = ['''떴다 떴다 비행기 날아라 날아라
          높이 높이 날아라 우리 비행기
         내가 만든 비행기 날아라 날아라
        멀리 멀리 날아라 우리 비행기''']
d['al'] = [1,2,3,4]
d['sr'] = sr
d
text1
pm.textAsvec(text1)
df = pm.textAsimport(text1)
text2 = '''떴다 떴다 비행기 날아라 날아라
        높이 높이 날아라 우리 비행기
        내가 만든 비행기 날아라 날아라
        멀리 멀리 날아라 우리 비행기'''
text2
df = pm.textAsimport(text2)

df_di = pd.DataFrame(di)
df_di
pm.catCodes(df_di)
df_di
#%%230829
file_path = 'D:\\.spyder-py3\\class file'
df_cctv = pd.read_csv(file_path + '\\서울시 자치구'
                      ' 년도별 CCTV 설치 현황_230630기준.csv', encoding='euc-kr') 

df_pop = pd.read_csv(file_path + '\\주민등록인구_20230828092210.csv')
new_df = pm.dfMerge(df_pop, df_cctv)
print(new_df)
new_df.info()
font_path = 'D:\\.spyder-py3\\class file\\NanumGothic.ttf'
pm.dfPlotScatt(new_df, font_path)
#%%230829
pwd
file_path = 'D:\\.spyder-py3\\class file'
tips = pd.read_csv(file_path + '\\tips.csv', sep=',')
gpd = tips.groupby(['day', 'smoker'])
for i, group in gpd:
    print(i)
group


df_cctv = pd.read_excel(file_path+'\\12_04_08_E_CCTV.xlsx')
df_pop = pd.read_csv(file_path+'\\주민등록인구_20230828092210.csv')



pm.PredictModelPlot(df_cctv, df_pop)

#%%230831
name = '삼성전자'
code = pm.findCodes(name)
df = web.DataReader(code, 'naver', start='2012-01-01', end='2023-08-29')

pm.prePlot(df, 'Close', '2023-04-30', 20)
pm. prophetPlot(df, 'Close', '2023-04-30', 107)


file_path = 'D:\\.spyder-py3\\class file'
df = pd.read_table(file_path + '\\temperature_ts_data', sep=',')
new_df = pm.timeDfAgg(df, 'timestamp', 'temperature', 'mean', 'size')
print(new_df)

#%%230904


file_path = 'D:\\download\\'
df_fish_main = pd.read_excel(file_path + '통합 식품영양성분DB_수산물_20230831.xlsx')
df_fish = df_fish_main.copy()
df_fish = pm.refine(df_fish)
top_ten = df_fish.loc[:,'단백질(micro)':'회분(micro)'].sum().sort_values(ascending=False)[:10].index

df_meve_main = pd.read_excel(file_path +'통합 식품영양성분DB_농축산물_20230831.xlsx')
df_meve = df_meve_main.copy()
df_meve = pm.refine(df_meve)
new_df_meve = df_meve.loc[:, '식품명']
for column in top_ten:
    if column in df_meve.columns:
        new_df_meve = pd.concat([new_df_meve, df_meve.loc[:, column]], axis=1)

result_df = pd.DataFrame()
for column in new_df_meve.columns[1:]:
    meve_sr = new_df_meve.loc[:,['식품명',column]].sort_values(by=column, ascending=False)[:3]['식품명'].values
    add_df = pd.DataFrame(meve_sr.reshape(1,3), index = [column], columns = ['1순위', '2순위', '3순위'])
    result_df = pd.concat([result_df, add_df])
print(result_df)

#%%230905

url = 'https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'
resp = requests.get(url)
soup = bs(resp.text, 'lxml')
rows = soup.select('div > ul > li')
etfs = pm.readweb(rows)
df = pd.DataFrame(etfs)

print(etfs.keys())
print(df)

#%%230908

file_path = 'D:\\.spyder-py3\\class file\\'
df = pd.read_csv(file_path+'survey.csv')
print(df.income.mean())
print(df.income.median())

male = df['income'][df['sex'] == 'm']
female = df['income'][df['sex'] == 'f']


# t 검정

print(stats.ttest_ind(male, female))	# t 검정 - H0 : 두 모집단의 평균 간에 차이가 없다.

# 등분산 검정 만족하는 t 검정
result = stats.ttest_ind(male, female) # 등분산 검정 만족하는 경우
alpha = 0.05
p_value = result[1]
print(p_value)
if p_value < alpha:
    print('귀무가설을 기각한다. 두 평균에 차이가 있습니다')
else:
    print('귀무가설을 채택한다. 두 평균에 차이가 없습니다')

# 등분산 검정 만족하지 못하는 t 검정
result2 = stats.ttest_ind(male, female, equal_var=False)  # 이분산의 방식 => 등분산 검정 만족하지 못함.
p_value = result2[1]
if p_value < alpha:
    print('두 평균에 차이가 있습니다')
else:
    print('두 평균에 차이가 없습니다')

# 등분산 검정
stat, p_value = stats.levene(male, female)	# Levelne 등분산 검정 - H0 : 두 집단의 분산은 같다
print(stat)
print(p_value)
alpha = 0.01
if p_value < alpha:
    print('귀무가설을 기각한다. 두 분산이 동일하지 않음. p-value :', p_value)
else:
    print('귀무가설을 채택한다. 두 분산이 동일함. p-value :', p_value)

# 정규성 검정
result = stats.shapiro(male) # 샤피로-윌크 검정 - H0 : 정규 분포를 따른다
alpha = 0.05
p_value = result[1]
if p_value < alpha:
    print('귀무가설을 기각한다. male 데이터는 정규 분포를 따르지 않는다.')
else:
    print('귀무가설을 채택한다. male 데이터는 정규 분포를 따른다.')

result = stats.shapiro(female) # 샤피로-윌크 검정 - H0 : 정규 분포를 따른다
alpha = 0.05
p_value = result[1]
if p_value < alpha:
    print('귀무가설을 기각한다. female 데이터는 정규 분포를 따르지 않는다.')
else:
    print('귀무가설을 채택한다. female 데이터는 정규 분포를 따른다.')

# 회귀분석
rv = stats.norm(2, 0.5) # norm:정규분포, loc:평균, scale:표준편차
x = np.arange(0, 4.1, 0.1)
y = rv.pdf(x)
plt.plot(x, y, lw=5) # lw : 그래프 선의 굵기
plt.grid()
plt.show()
print(rv.cdf(1.7)) # 그래프의 왼쪽에서부터 x=1.7 까지의 넓이 (=확률)
print(rv.isf(1-0.27425311775007355)) # 그래프의 오른쪽을 기준으로 넓이 a를 만족하는 x축값. -> 전체넓이는 1

# 넓이 0.9인 그래프의 가운데 구간을 구해보자

print(f'시작 x값 : {rv.isf(0.95)}')
print(f'끝 x값 : {rv.isf(0.05)}')