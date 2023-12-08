#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:20:50 2022

@author: wangshuyou
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import kruskalwallis #無母數檢定
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

df_Full_sati = pd.read_excel('/Users/wangshuyou/中山招生策略組RA/EMI分析/110-2/110＿中英語課程教學意見＿學習成效比較＿全校.xlsx', sheet_name='org_滿意度')[['學期','課程代碼','教師名稱','屬性','全英','mean']]

df_Full_sati_C = df_Full_sati[df_Full_sati['全英'] == 0]
df_Full_sati_E = df_Full_sati[df_Full_sati['全英'] == 1]

# 1. 中文班平均高的老師

T_mean_C = df_Full_sati_C[['教師名稱','mean']].groupby('教師名稱').agg({'教師名稱':len,'mean':np.mean})
mean_higher = T_mean_C[(T_mean_C['mean'] >= 6.16)& (T_mean_C['教師名稱'] > 30)].sort_values('mean',ascending=True)
print(len(T_mean_C))
print(mean_higher.head(10))
print('平均大於6.5分人數：',len(mean_higher))

# 2. 英文授課平均高的老師

T_mean_E = df_Full_sati_E[['教師名稱','mean']].groupby('教師名稱').agg({'教師名稱':len,'mean':np.mean})
mean_higher_E = T_mean_E[(T_mean_E['mean'] >= 6.2)& (T_mean_E['教師名稱'] > 30)].sort_values('mean',ascending=False)
print(len(T_mean_E))
print(mean_higher_E.head(10))
print('平均大於6.5分人數：',len(mean_higher_E))


# 計算中英文授課個別老師被評量次數及平均
# 中文
T_name1C = df_Full_sati_C.groupby(['教師名稱'])['mean'].agg({'count' , np.mean})
T_name1E = df_Full_sati_E.groupby(['教師名稱'])['mean'].agg({'count', np.mean})
print(T_name1C[T_name1C['count'] > 600])
print(T_name1C[T_name1C['mean'] < 5].count())
print(T_name1C['mean'].nsmallest(1))

# 英文
print(T_name1E['count'].nlargest(1))
print(T_name1E[T_name1E['count'] > 159])
print(T_name1E[T_name1E['mean'] < 5].count())
print(T_name1E['mean'].nsmallest(1))

g = sns.relplot('count','mean', data = T_name1C, kind = 'scatter', hue = 'count')
g.fig.suptitle('教學滿意度＿中文授課')
plt.xlabel('填答人數')
plt.ylabel('平均分數')
plt.axhline(y = 5, color='red')
plt.show()

g1 = sns.relplot('count','mean', data = T_name1E, kind = 'scatter', hue = 'count')
g1.fig.suptitle('教學滿意度＿英文授課')
plt.xlabel('填答人數')
plt.ylabel('平均分數')
plt.axhline(y = 5, color='red')
plt.show()

sati_test1 = df_Full_sati[df_Full_sati['全英'] == 0]['mean'].values
sati_test2 = df_Full_sati[df_Full_sati['全英'] == 1]['mean'].values
f, p = kruskalwallis(sati_test1,sati_test2)
if p < 0.1:
    print('顯著')
else:
    print('none')


# 3. 第一學期

df_Full_sati_1 = df_Full_sati[df_Full_sati['學期'] == 1]
df_Full_sati_2 = df_Full_sati[df_Full_sati['學期'] == 2]


df_Full_sati_1_0 = df_Full_sati_1[df_Full_sati_1['全英']==0]['mean'].values
df_Full_sati_1_1 = df_Full_sati_1[df_Full_sati_1['全英'] == 1]['mean'].values
f, p = kruskalwallis(df_Full_sati_1_0,df_Full_sati_1_1)
if p < 0.05:
    print('顯著')
else:
    print('none')
    
# 4. 第二學期


df_Full_sati_2_0 = df_Full_sati_2[df_Full_sati_2['全英']==0]['mean'].values
df_Full_sati_2_1 = df_Full_sati_2[df_Full_sati_2['全英'] == 1]['mean'].values
f1, p1 = kruskalwallis(df_Full_sati_2_0,df_Full_sati_2_1)
if p1 < 0.05:
    print('顯著')
else:
    print('none')
