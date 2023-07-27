# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import folium

#%%230727
def change_list(filename):
    lines_list = []
    result_list = []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        lines_list.append(list(map(int, line.strip().split())))
    for i in range(len(lines_list[0])):
        save_list = []
        for j in lines_list:
            save_list.append(j[i])
        result_list.append(save_list)
    return result_list

def trans_matrix(file_path):
    with open(file_path, 'r') as f:
        matrix = [list(map(int, line.strip().split())) for line in f]
        return [list(row) for row in zip(*matrix)]
    
def change2_list(filename):
    with open(filename) as f:
        lines_list = [list(map(int, line.strip().split())) for line in f.readlines()]
    result_list = [[j[i] for j in lines_list] for i in range(len(lines_list[0]))]
    return result_list
def price_profit(listname):
    if not listname or len(listname) < 2:
        return 0
    min_price = listname[0]
    max_profit = 0 
    for price in listname:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    max_price = min_price + max_profit
    return min_price, max_price
def MarkerMap(data_dict):   #pip install folium 프롬프트로 설치
    maps = folium.Map(location=[37.5602, 126.982], zoom_start=7, tiles='cartodbpositron')
    for i in data_dict:
        name = i
        lat_ = data_dict[i][0]
        long_ = data_dict[i][1]
        folium.CircleMarker([lat_, long_], radius = 4, popup = name, color = 'red', fill_color = 'red').add_to(maps)
    return maps
#%% 계산기
result = 0

def add(num):
    global result
    result += num
    return result

#%% class를 이용한 계산기
class Calculator:
    def __init__(self):     # 생성자,   메서드
        self.result = 0
    def add(self, num):     # 메서드
        self.result += num
        return self.result
#%%
def find_data(dataset):
    key = ['rank', 'movieNm', 'openDt', 'salesAmt']
    result_data = {}
    for i, j in dataset.items():
        if i in key:
            result_data[i] = j
    return result_data

def index_data(data):
    for i in data['boxOfficeResult']['dailyBoxOfficeList']:
        result_data_dict = find_data(i)
        print(result_data_dict)