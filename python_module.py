# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import math

#%%230731

class MyStr:
    def __init__(self, s):
        self.s = s
    def __add__(self, b):
        print('왼쪽에 있을때만 사용해 주세요.')
        return self.s + b
    def __radd__(self, b):
        print('오른쪽도 사용 가능 합니다.')
        return b + self.s

class FourCal2:
    def __init__(self, first, second):
        self.setdata(first, second)
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def __add__(self, a):
        return self.first + self.second + a
    def __radd__(self, a):
        return a + self.first + self.second   
    def __mul__(self, a):
        return self.first * self.second * a
    def __rmul__(self, a):
        return  a * self.first * self.second 
    def __sub__(self, a):
        return self.first - self.second - a
    def __rsub__(self, a):
        return a - self.first - self.second
    def __truediv__(self, a):
        return self.first / self.second / a
    def __rtruediv__(self, a):
        return a / self.first / self.second
    
class A:
    def __repr__(self):
        return('class A print')
class B:
    def __str__(self):
        return('class B print')
class C(A):
    def __str__(self):
        return('class C print')
    
class StringRepr:
    def __repr__(self):
        return 'repr called'

    def __str__(self):
        return 'str called'
    
class StringRepr2:
    def __str__(self):
        return 'str called'
    
class StringRepr3:
    def __repr__(self):
        return 'repr called'
    
class StringRepr4:
    def __init__(self, i = 10):
        self.i = i
    def __repr__(self):
        return 'pm.StringRepr4(100)'
    
class MyClass7:
    def __init__(self, x, y ):
        self.x = x
        self.y = y
    def __repr__(self):
        return f'MyClass(x = {self.x}, y = {self.y})'
    
class Accumulator:
    def __init__(self):
        self.sum = 0
    def __call__(self, *args):
        self.sum += sum(args)
        return self.sum
    
class A:
    def __call__(self, v):
        return v
class B:
    def func(self, v):
        return v
def check(func):
    if callable(func):
        print('callable')
    else:
        print('not callable')
        
class Factorial:
    def __call__(self, n):
        return self.factorial(n)
    def factorial(self, n):
        result = 1
        if n == 0:
            return result
        else:
            for i in range(1, n+1):
                result *= i
        return result
    
class MyStr2:
    def __init__(self, s):
        self.s = s 
    def __repr__(self):
        return f"My_Str('{self.s}')"
    def __add__(self, n):
        return f"My_Str('{self.s + n}')"
    def __sub__(self, n):
        self_s_list = self.s.split(' ')
        num = self_s_list.index(n)
        return MyStr2(' '.join(self_s_list[:num] + self_s_list[num+1:]))
    
class Test_no_getitem:
    def __init__(self):
        print('생성자 __init__을 호출하였습니다.')
        self._numbers = [ n for n in range(1, 11)]
    def __getitem__(self, index):
        print('__getitem__을 호출하였습니다.')
        return self._numbers[index]
    
class MyList:
    def __init__(self, items):
        self.items = items
    def __getitem__(self, x):
        if isinstance(x, int):
            return self.items[x]
        elif isinstance(x, str):
            return self.items.index(x)
        else:
            raise TypeError('잘못된 형식입니다.')
            
class Square:
    def __init__(self, items):
        self.items = [ i**2 for i in range(items+1) ]
    def __getitem__(self, indexs):                      # 반복자 for문 사용이나 형변환시 오류발생 고려
        if len(self.items) >= indexs >= 0:
            return self.items[indexs]
        raise IndexError('list index out of range')
        
class Person:
    def __init__(self, name, phone=None):
        self.name = name
        self.phone = phone
    def __repr__(self):
        return f'<Person {self.name} {self.phone} >'
class Employee(Person):
    def __init__(self, name, phone, position, salary):
        #Person.__init__(self, name, phone)
        #super(Employee, self).__init__(name, phone)
        super().__init__(name, phone)

        self.position = position
        self.salary = salary
        
class Employee2(Person):
    def __init__(self, name, phone, position, salary):
        super().__init__(name, phone)
        self.position = position
        self.salary = salary
    def __repr__(self):
        s = super().__repr__()
        return s + f'Employee {self.name} {self.phone} {self.position} {self.salary}'
    
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def area(self):
        return 0
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
    def __repr__(self):
        return f'x = {self.x}, y = {self.y}'
class Circle(Point):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.radius = r
    def area(self):
        return math.pi * self.radius * self.radius
    def __repr__(self):
        return f'{super().__repr__()}, radius = {self.radius}'
class Cylinder(Circle):
    def __init__(self, x, y, r, h):
        super().__init__(x, y, r)
        self.height = h
    def area(self):
        return 2 *Circle.area(self) + 2*math.pi*self.radius * self.height
    def volum(self):
        return super().area()*self.height
    def __repr__(self):
        return f'{super().__repr__()} height={self.height}'   
    
class Job:
    def __init__(self, position, salary):
        self.position = position
        self.salary = salary
    def __repr__(self):
        return f'Job position = {self.position} salary = {self.salary}'
class Employee3(Person, Job):
    def __init__(self, name, phone, position, salary):
        Person.__init__(self, name, phone)
        Job.__init__(self, position, salary)
        
    def __repr__(self):
        return Person.__repr__(self) + Job.__repr__(self)    
    
    
    