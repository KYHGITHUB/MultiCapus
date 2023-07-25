# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""
import python_module as pm

#%% 230713
'''
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
'''
#%%
#test_string = '''Humans are odd. They think order and chaos are somehow opposites and try to control what won't be. But there is grace in their failings.'''
'''print(pm.change_string(test_string))
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
'''
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

