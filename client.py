# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""
import python_module as pm


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



















