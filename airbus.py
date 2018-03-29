# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:02:23 2018

@author: aveissei
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

data = pd.read_csv('C://Users//aveissei//Desktop//Airbus//A320_7751_F0001.csv', delim_whitespace=True)

data = data.iloc[3:, :]
data['GMT'] = pd.to_datetime(data['GMT'])
data = data.set_index('GMT')
data = data.iloc[:, 1:]

data = data.astype('int64')

drop_lst = []

def transformer_liste(tableau):
    tempo=[]
    for i in tableau:
        tempo.append(i[:3]+i[4:])
    tempo = list(set(tempo))
    return list(set(tempo))  



to_drop = []
to_keep = []
for t,col in enumerate(list(data.columns)):
    col_recons = col[:3]+col[4:]
    tempo = transformer_liste(to_keep)
    if col_recons in tempo:
        to_drop.append(col)
    else:
        to_keep.append(col)

data = data.drop(labels=to_drop, axis=1)        

"""
def transformer_liste2(tableau):
    tempo=[]
    for i in tableau:
        tempo.append(i[:4]+i[5:])
    tempo = list(set(tempo))
    return list(set(tempo))  

to_drop = []
to_keep = []
for t,col in enumerate(list(data.columns)):
    col_recons = col[:4]+col[5:]
    tempo = transformer_liste2(to_keep)
    if col_recons in tempo:
        to_drop.append(col)
    else:
        to_keep.append(col)

data = data.drop(labels=to_drop, axis=1) 
"""
for c in data.columns:
    if (data[c].sum() == 0) or (data[c].sum() == data.shape[0]):
        drop_lst.append(c)

data = data.drop(labels=drop_lst, axis=1)

data_finale = data.copy()
data_finale = np.zeros((data.shape[0],data.shape[1]))

for colonne in tqdm(range(data.shape[1])):
    for ligne in range(data.shape[0]-1):
        if (data.iloc[ligne, colonne] != data.iloc[ligne+1, colonne]):
            if data.iloc[ligne, colonne] == 0:
                data_finale[ligne,colonne]=1
            else:
                data_finale[ligne,colonne]=1

frames = [pd.DataFrame(data.index),pd.DataFrame(data_finale)]

result = pd.concat(frames,axis=1)

my_data = result.set_index(result['GMT'])

my_data = my_data.drop(labels=['GMT'], axis=1)
grouped = my_data.resample('30S', how='sum')
  
grouped.columns = list(data.columns)

grouped[grouped>1]=1

def correlation(df,colonne1,colonne2):
    filtered = df[df[colonne1]==1][colonne2]
    if len(filtered)>0:
        coef = round(np.sum(filtered)/len(filtered),1)
    else:
        coef = 0
    return coef

def cross_cor(df,colonne1,colonne2):
    coef1 = correlation(df,colonne1,colonne2)
    coef2 = correlation(df,colonne2,colonne1)
    return coef1*coef2

list_col = list(grouped.columns)
mat_cor = pd.DataFrame(np.zeros((len(list_col), len(list_col))),columns = list_col,index=list_col)

for i,col1 in enumerate(list_col):
    print('avancement ' + str(round(100*i/len(list_col),0)))
    for j,col2 in enumerate(list_col):
        coef = cross_cor(grouped,col1,col2)
        mat_cor.iloc[j,i] = coef

X = range(len(data))

s = 0.4
for i,col1 in enumerate(list_col):
    flag = True
    current_mat = mat_cor[col1]
    current_mat = current_mat.sort_values(ascending=False)
    current_mat = current_mat[current_mat>=s]
    current_index = list(current_mat.index)
    for p in current_index:
        if (sum(data[col1]==data[p])==len(X) and col1 != p):
            current_index.remove(p)
    if len(current_index) < 2 or len(current_index)>10:
        flag = False
    if flag == True:    
        for k,col in enumerate(current_index):
            Y = np.array(data[col]+k)
            print(col)
            if col ==col1:
                plt.plot(X,Y,label=col,linestyle='dashed')
            else:
                plt.plot(X,Y,label=col)
        plt.legend()
        plt.savefig('C://Users//aveissei//Desktop//Airbus//fig2//'+str(col1)+'.jpg', format="jpg")
        #plt.show()
        plt.close()
  