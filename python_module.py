# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 15:24:17 2023

@author: rnjsd
"""
import pandas as pd
import unicodedata

#%%230905

def readweb(rows):
    etfs = {}
    for row in rows:
        for index in range(len(row.text.split('\n'))):

            try:
                row_text = row.text.split('\n')[index]
                start = row_text.rfind('(')
                end = row_text.rfind(')')
                if start > 2:
                    new_row = unicodedata.normalize("NFKD", row_text)
                    if '|' in new_row[start+1:end].replace(':', ''):
                        etf_name = [new_row[:start-1]]
                        etf_market = [new_row[start+1:end].replace(':', '').split('|')[0]]
                        etf_ticker = [new_row[start+1:end].replace(':', '').split('|')[-1]]
                    else:
                        etf_name = [new_row[:start-1]]
                        etf_market = [' '.join(new_row[start+1:end].replace(':', '').split(' ')[:-1])]
                        etf_ticker = [new_row[start+1:end].replace(':', '').split(' ')[-1]]
                        
                    if (len(etf_ticker) > 0) & (len(etf_market) > 0) & (len(etf_name) > 0):
                        etfs[etf_ticker[0]] = [etf_market[0], etf_name[0]]            
            except AttributeError as err:
                pass
    return etfs


