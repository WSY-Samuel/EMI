#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 18:34:15 2022

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



df_Full_self = pd.read_excel('/Users/wangshuyou/中山招生策略組RA/EMI分析/110-2/110＿中英語課程教學意見＿學習成效比較＿全校.xlsx', sheet_name='org_成效')[['SEM','CRSNO','T_NAME','屬性','全英','mean']]

df_Full_self_C = df_Full_self[df_Full_self['全英'] == 0]
df_Full_self_E = df_Full_self[df_Full_self['全英'] == 1]

# 1. 中文班平均高的老師
T_mean_C = df_Full_self_C[['T_NAME','mean']].groupby('T_NAME').agg({'T_NAME':len,'mean':np.mean})
mean_higher = T_mean_C[(T_mean_C['mean'] >= 6.03)& (T_mean_C['T_NAME'] > 30)].drop('指導教授').sort_values('mean',ascending=False)
print(len(T_mean_C))
print(mean_higher.head(10))
print('平均大於6.5分人數：',len(mean_higher))

# 2. 英文授課平均高的老師
T_mean_E = df_Full_self_E[['T_NAME','mean']].groupby('T_NAME').agg({'T_NAME':len,'mean':np.mean})
mean_higher_E = T_mean_E[(T_mean_E['mean'] >= 6)& (T_mean_E['T_NAME'] > 30)].sort_values('mean',ascending=False)
print(len(T_mean_E))
print(mean_higher_E.head(10))
print('平均大於6.5分人數：',len(mean_higher_E))


# 計算中英文授課個別老師被評量次數及平均
# 中文
T_name1C = df_Full_self_C.groupby(['T_NAME'])['mean'].agg({'count' , np.mean})
T_name1E = df_Full_self_E.groupby(['T_NAME'])['mean'].agg({'count', np.mean})
print(T_name1C[T_name1C['count'] > 500])
print(T_name1C[T_name1C['mean'] < 5].count())
print(T_name1C['mean'].nsmallest(1))

# 英文
print(T_name1E['count'].nlargest(1))
print(T_name1E[T_name1E['count'] > 140])
print(T_name1E[T_name1E['mean'] < 5].count())
print(T_name1E['mean'].nsmallest(1))

g = sns.relplot('count','mean', data = T_name1C, kind = 'scatter',hue = 'count')
g.fig.suptitle('自評學習成效＿中文授課')
plt.xlabel('填答人數')
plt.ylabel('平均分數')
plt.axhline(y = 5, color='red')
plt.show()

g1 = sns.relplot('count','mean', data = T_name1E, kind = 'scatter', hue = 'count')
g1.fig.suptitle('自評學習成效＿英文授課')
plt.xlabel('填答人數')
plt.ylabel('平均分數')
plt.axhline(y = 5, color='red')
plt.show()
# =============================================================================
# 
# df_Full_self_1 = df_Full_self[df_Full_self['SEM'] == 1]
# df_Full_self_2 = df_Full_self[df_Full_self['SEM'] == 2]
# 
# # 1. 第一學期
# 
# df_Full_self_1_0 = df_Full_self_1[df_Full_self_1['全英']==0]['mean'].values
# df_Full_self_1_1 = df_Full_self_1[df_Full_self_1['全英'] == 1]['mean'].values
# f, p = kruskalwallis(df_Full_self_1_0,df_Full_self_1_1)
# if p < 0.1:
#     print('顯著')
# else:
#     print('none')
# 
# # 2. 第二學期
# df_Full_self_2_0 = df_Full_self_2[df_Full_self_2['全英']==0]['mean'].values
# df_Full_self_2_1 = df_Full_self_2[df_Full_self_2['全英'] == 1]['mean'].values
# f1, p1 = kruskalwallis(df_Full_self_2_0,df_Full_self_2_1)
# if p1 < 0.1:
#     print('顯著')
# else:
#     print('none')
# 
# =============================================================================
