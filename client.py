# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""

import python_module as pm


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

c2 = pm.B()
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





