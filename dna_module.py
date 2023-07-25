# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:11:37 2023

@author: rnjsd
"""

#%%230725
def dna_dvi(dna):
    dna_list = []
    for i in range(len(dna)+1):
        if i % 3 == 0:
            if i != 0:
                dna_list.append(dna[i-3:i])
    dna_list = dna_list[1:-1]
    return dna_list

def dna_disc(dna):
    print(dna_dvi(dna))
    result = False
    if len(dna) % 3 == 0:
        if dna.startswith('ATG'):
            if dna.endswith('TAA') or dna.endswith('TGA') or dna.endswith('TAG'):
                if dna_dvi(dna).count('TAA') == 0 and  dna_dvi(dna).count('TGA') == 0 and dna_dvi(dna).count('TAG') == 0:
                    result = True
                    return result                  
    return result