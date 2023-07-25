# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:15:46 2023

@author: rnjsd
"""
import os
import glob

#############230725##########

def search_file(file_name, keyword):
    result = []
    with open(file_name, 'r', encoding ='utf-8') as f:
        for lines in f:
            if keyword in lines:
                result.append(lines.strip())
        return result

def search_folder(folder, keyword):
    search_result = {}
    for root, dirs, files in os.walk(folder):   #현재 디렉토리, 하위 디렉토리, 파일 이름
        for file in files:
            file_path = os.path.join(root, file)      
            if file.endswith('.txt') or file.endswith('.py'):
                search_result[file_path] = search_file(file_path, keyword)
                if search_result[file_path] == []:
                    del search_result[file_path]
    for file_path in search_result.keys():
        print(file_path)
    path = input('원하는 경로를 입력해주세요 :')
    #output.clear()   #코랩에서만 작동
    #os.system('cls') #스파이더에서 작동안함 -> 파워쉘에서 작동함    
    print(f'경로 : {path}')
    print('='*80)
    for i in search_result[path]:
        print(i)

#%%
def GetFileSentence(file_list, search_sentence, s_path):    # s_path : __import__()
    file_dict = {}
    for filename in file_list:
        sentence_list = []
        with open(filename, encoding='utf-8') as f:
            lines = f.readlines()
        for item in lines:
            if search_sentence in item:
                sentence_list.append(item)
        if sentence_list :
            file_dict[filename[len(s_path):]] = sentence_list
    return file_dict

def FileCollection(L_name, file_types):
    #s_path = str(__import__(L_name)).split('\'')[3][:-11]
    s_path = '\\'.join(str(__import__(L_name)).split()[-1].split('\'')[1].split('\\')[:-1])+'\\'
    dummy_arg = s_path + '**\\\*.'+file_types
    print(dummy_arg)
    file_lists = glob.glob(dummy_arg, recursive=True)
    return (file_lists, s_path)

def print_result(fdr):
    for i in fdr:
        print(i)
        print('==='*10)
        for j in fdr[i]:
            print('-   ', j.strip())
        print()
            
#%%
