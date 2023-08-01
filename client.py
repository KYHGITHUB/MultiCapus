# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""

import python_module as pm



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


