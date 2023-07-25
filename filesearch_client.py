# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:16:44 2023

@author: rnjsd
"""
#############230725################
import filesearch_module as fsm

(file_list, s_path) = fsm.FileCollection('pandas', 'py')
#print(s_path)
#print(file_list)

f_dict = fsm.GetFileSentence(file_list, 'index', s_path)
#print(f_dict)

fsm.print_result(f_dict)
