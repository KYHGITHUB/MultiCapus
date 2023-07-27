# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:22:47 2023

@author: rnjsd
"""
import python_module as pm
import matplotlib.pyplot as plt
import requests # 웹서버 가지고 오는 라이브러리
import sys_path_add
from module_test import module_test as mt


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