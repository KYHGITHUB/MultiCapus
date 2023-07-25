# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:13:23 2023

@author: rnjsd
"""

a = 3
if a > 2:
    print('a는 2보다 큽니다')

a = 0
if a > 2:
    print('a는 2보다 큽니다')
else:
    print('a는 ', a,'입니다')

names = input('이름을 입력해주세요. : ')
if names == '권용현' :
    print('본인입니다. ')
else:
    print('본인이 아닙니다. ')
    
mon = int(float(input('아이가 태어난 지 몇 개월입니까? : ')))
if mon == 1 :
    print('결핵 예방접종 대상자입니다.')
if 1 <= mon <= 2 :
    print('B형간염 예방접종 대상자입니다.')
if 2 <= mon <= 6 :
    print('파상풍 예방접종 대상자입니다.')
if 2 <= mon <= 15 :
    print('폐렴구균 예방접종 대상자입니다.')
if 16 <= mon :
    print('예방접종 대상자가 아닙니다 ')

age = 6
if age < 5 :
    print('ok')
elif age > 5 and age < 10 :
    print('ok, good')
else:
    print('no')
    
if not age > 10 :
    print('ok')
    
fash = '맨발'
if not fash == '샌들양말' :
    print('모두 좋습니다.')
    
if 4 in [1, 2, 3]:
    print('ok')
    
x_list = [1, 2, 3]
x_list

new_town_list = ['경기도', '서울', '대전', '광주']
if '서울' in new_town_list:
    print('ok')
    
num1 = 10
num2 = 14
num3 = 21
if num1 > num2 and num1 > num3 :
    print('최댓값은 ', num1)
elif num2 > num3 :
    print('최댓값은 ', num2)
else:
    print('최댓값은 ', num3)
    
year = 2020
cond1 = (year%4) == 0
cond2 = (year%100) != 0
cond3 = (year%400) == 0
case1 = cond1 and cond2 or cond3
if case1 :
    print('leap year')
else:
    print('not leap year')
    
score = int(float(input('점수를 입력하세요. :')))
if 90 <= score :
    print('학점 : A')
elif 80 <= score :
    print('학점 : B')
elif 70<= score :
    print('학점 : C')
elif 60<= score :
    print('학점 : D')
else:
    print('학점 : F')
    
x = int(input('값을 입력하세요. :'))
if x <= 0:
    print('자연수가 아닙니다.')
elif x%2 == 0:
    print('짝수입니다.')
else:
    print('홀수입니다')
    
height = float(input('키를 입력해주세요. '))
weight = float(input('몸무게를 입력해주세요. :'))
bmi = weight / (height/100)**2

if bmi < 18.5:
    print('저체중 입니다')
elif bmi <= 22.9:
    print('정상 체중 입니다')
elif bmi <= 24.9:
    print('과체중 입니다')
elif bmi <= 29.9:
    print('비만 입니다')
else:
    print('고도비만 입니다')
    
s = 'Strings in Python'
print(s)
print(s[0])
print(s[::])
print(s[0:17:1])
print(s[:len(s):])

print(x_list[1:2])
print(x_list[0:1])
print(s[::])
wind = 'No matter how the wind howls, the mountain cannot bow to it.'
print(wind)
print(len(wind))
print(wind[12])
print(wind[::3])
print(wind[-60:-1])
sample_list = [10, 20, 30, 40, 50, 60]
print(sample_list[0], sample_list[2], sample_list[3])
print(sample_list[:20]) #len(sample_list) 보다 크면 최대갯수로 봐줌
print(sample_list[::])
sample_list = [10, [12, wind]]
print(sample_list)
print(sample_list[1][1][18:22])
for i in sample_list:
    print(i)
print(range(6))
print(list(range(12,20)))
x = range(0, 100000000)
for i in x[:3]:
    print(i)
print(range(0, 10, 2))  # start = 0, stop = 10, step = 2
print(list(range(0, 10, 2)))
print(len(sample_list))
for i in range(len(sample_list)):
    print(i, sample_list[i])

for i in [10, 20, 30] :
    print(i)
    
for i in range(len('test_sentence')) :
    print(i)
    
test_list = [1, 3, 'test', wind, sample_list]
print(test_list)
test_list

color = ['black', 'green', 'red', 'puple', 'blue', 'white']
for i in color :
    print(i)
    
word = ['computer', 'game', 'youtube', 'drive', 'manager', 'zoom']
for i in word :
    print(i, len(i))
    
num = [1, 2, 3, 4, 5]
for i in num[1:] :
    print(i)
    
for i in range(0, 11) :
    print(i)
    
for i in range(5, 16) :
    print(i)
    
for i in range(0, 22, 2) :
    print(i)
    
for i in range(-100, 101, 4) :
    print(i)
    
x = 0
for i in range(1, 101) :
    x = x + i
    if i == 100 :
       print(x)
       
x = 0
for i in range(1, 101) :
    x += i
print(x)                  #반복문 끝내려면 들여쓰기를 없애면 된다

x = 1
for i in range(1, 11) :
    x *= i
print(x)

for i in range(1, 10) :
 print('2 *', i, '=', 2 * i)
#%%
for i in range(2, 10) :
    for j in range(1, 10) :
        print(i, '*', j, '=', i * j, end = '    ')
    print()

#%%
for i, j in enumerate(test_list) :
    print(i, j)
    
n = 0
while n < 5 :
    n += 1
    print(n)
    #%%
i = 2
num = int(input('숫자를 입력하세요. : '))
while num % i != 0 :
    i += 1
if i == num :
    print('소수입니다')
else:
    print('소수가 아닙니다')
#%%