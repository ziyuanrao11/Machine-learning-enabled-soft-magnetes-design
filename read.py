# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:04:30 2022

@author: z.rao
"""


import pandas as pd
import re
#%%read the data
target = pd.read_excel('HEAs.xlsx').drop('Source',axis=1)
alloys=target['Alloys']
comp=pd.DataFrame(index=range(len(alloys)))
# print(comp.columns.values)
#%%calculate the compositions in each alloys
for i in range(len(alloys)):
    split=re.split('(\d+(?:\.\d+)?)', alloys[i])
    split.pop(-1)
    print(i)
    print(split)
    for j in range(int(len(split)/2)):
        element=split[2*j]
        content=float(split[2*j+1])
        comp.loc[i,element]=content
        # if element in comp.columns.values:
        #     comp.loc[i,element]=content
        # else:
        #     comp[element]=0   
comp=comp.fillna(0)
comp.to_excel('Comp.xlsx')
print(comp)
#%%save the data
All=pd.concat([comp,target],axis=1)
All=All.drop('Alloys',axis=1)
All.to_excel('all_data.xlsx')
        
 
