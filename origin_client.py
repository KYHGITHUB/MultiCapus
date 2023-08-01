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