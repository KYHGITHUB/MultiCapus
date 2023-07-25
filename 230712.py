# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:46:53 2023

@author: rnjsd
"""
#%%
num = 103
i = 2
flag = 1
while i < num :
    #i += 1
    res = num % i
    if res == 0 :
        print(i)
        flag = 0
        break    # 상위 반복문으로 빠져나옴
        #continue  # 속한 반복문 첫줄로 이동 ( 다음줄 코드 생략)
    i += 1

if flag :
    print('소수 입니다.')
else:
    print('소수가 아닙니다.')
#%%
for i in range(10) :
    if 3 < i < 6 :
        continue
        break
    print(i)
#%%
i = 0
while i < 10 :
    i += 1
    if 8 > i > 6 :
        break
    print(i)
#%%
def prim(number) :
    for i in range(2, number) :
        if number % i == 0 :
            return (False)
    return (True)
print(prim(7))
#%%
passwds = input('패스워드를 입력해 주세요. : ')
r_passwds = '1234567'
while passwds != r_passwds :
    if passwds != r_passwds :
        print('정확한 비밀번호를 입력해주세요')
        passwds = input('패스워드를 정확하게 입력해 주세요. : ')
    else :
        break
print('패스워드 입력에 성공하셨습니다')
#%%
names = input('당신의 이름을 입력하세요. q를 입력하면 종료합니다. : ')
stop = 'q'
while names != stop :
    print(names)
    names = input('당신의 이름을 입력하세요. q를 입력하면 종료합니다. : ')
print(stop)
#%%   # 공부할것
num = int(input('2진수로 변환하기 위한 숫자를 입력하세요. :'))
mum = ''
while num > 0   :
    mum = str(num % 2) + mum
    num //= 2
print(mum)
#%%
num = int(input('예상 숫자를 입력하세요. : '))
correct = 50
while num != correct :
    if num > correct :
        print('DOWN')
    elif num < correct :
        print('UP')
    num = int(input('예상 숫자를 다시 입력하세요. : '))
print('정답')
#%%
se = ''' Humans are odd. They think order and chaos are somehow opposites and try to control what won't be. But there is grace in their failings. '''
print(se)
print(se.isalpha()) #영어인지 확인하는 명령어
print(se.isnumeric()) #숫자인지 확인하는 명령어
print('2'.isnumeric())
print(se.lower()) #소문자로 변환
print(se.upper()) #대문자로 변환
print(se.strip() ) #문장의 앞뒤로 괄호 안에 해당하는 부분을 삭제. 기본값은 space
print(se.replace('Humans', 'We')) #왼쪽값을 오른쪽값으로 변환
result = se.split()
print(result)
result1 = '\\'.join(result)
print(result1)
print(se.replace('.', '\n'))
se0 = se.strip()
se1= '/'.join(se0.split())
se2 = se1.replace(' ','') 
se3 = se2.replace('.','')
se4 = se3.replace("'",'')
print(se4)
new_se = '/'.join(se.strip().replace('.', '').replace("'", '').split())
print(new_se)
print(se.find(' ', 2))
print(se[:3])
print(se[101])
print(se.index('u'))
print(se.find('z')) # find와 index의 차이 : 존재 하지 않는 값을 관측 시도시 find는 알려주고 index는 알려주지 않음
#%%
sentence = input('문장을 입력하세요. q입력시 종료합니다. :')
t = 'q'
while sentence != t :
    print('이 문장은', len(sentence.split()), '어절 입니다. ')
    sentence = input('문장을 입력하세요. q입력시 종료합니다. :')
print(t)
#%%
level = 10
while level < 200 :
    level += 1
    if level % 2 != 0 :
        continue
    print(level)
#%%
level = 10
while level < 200 :
    level += 1
    if level % 169 == 0 :
        break
print(level)
#%%
for item in range(2, 4):
    for each in range(2, 6):
        print('%d X %d = %d' %(item, each, item * each))
#%%
dir() # 사용중인 변수확인
i = 1
while i < 14:
    print(f'제{i}의아해가무섭다고그리오')
    i += 1
#%%
print(round(8000 / 1.1, 1))
y = lambda para1 : 3 * para1
print(y(4))
add = lambda a, b : a + b
print(add(2, 3))
print(se[:10])
short = lambda x : x[:10]
print(short(se))
def calculator(a, b):
    return a + b, a - b, a * b, a / b
print(calculator(12, 3))
print(type(calculator(12, 3)))
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
se
result = se.replace(' ', \n)
result
