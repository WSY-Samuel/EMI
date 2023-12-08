# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.mstats import kruskalwallis #無母數檢定
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# 解決mac圖片中文顯示問題
plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

df_Full_sati = pd.read_excel('/Users/wangshuyou/中山招生策略組RA/EMI分析/110-2/110＿中英語課程教學意見＿學習成效比較＿全校.xlsx', sheet_name='org_滿意度')[['學期','課程代碼','教師名稱','屬性','全英','mean']]

#檢查缺失值
print(df_Full_sati.isna().any())

sati_test1 = df_Full_sati[df_Full_sati['全英'] == 0]['mean'].values
sati_test2 = df_Full_sati[df_Full_sati['全英'] == 1]['mean'].values

sati_table = pd.pivot_table(df_Full_sati, values ='mean', index ='屬性', 
                         aggfunc = [len, np.mean , np.std]
                         ) 
print(sati_table)
# =============================================================================
# 常態檢定

# shapiro 檢定適合用在小樣本 小於50
# =============================================================================
# print(stats.shapiro(sati_test1))
# print(stats.shapiro(sati_test2))
# =============================================================================

# Kolmogorov–Smirnov Test: 基於累計分布函數，用以檢驗兩個經驗分布是否不同或一個經驗分布與另一個理想分布是否不同
# 1. 一般班
sati_loc1, sati_scale1 = stats.norm.fit(sati_test1)
sati_n_1 = stats.norm(loc=sati_loc1, scale=sati_scale1)
plt.hist( sati_test1,  
          bins=np.arange(sati_test1.min(), sati_test1.max()+0.2, 0.2), 
          rwidth=0.5 )

x = np.arange(sati_test1.min(), sati_test1.max()+0.2, 0.2)
plt.plot(x, 11000*sati_n_1.pdf(x))
plt.title('一般班平均教學滿意度')
plt.show()

t_11,pv11 = stats.kstest(sati_test1,sati_n_1.cdf)
if pv11 > 0.01:
   print(f'常態分配,P_value : {pv11}')
else:
   print(f'非常態分配,P_value : {pv11}')
# 其檢測之虛無假設 (H0):本變項之樣本群為常態分佈。 因此如果 p < 0.05，則推翻虛無假設，表示為非常態分佈; 如果是 p > 0.05，則接受虛無假設，表示為常 態分佈。

# 2. 全英班
sati_loc2, sati_scale2 = stats.norm.fit(sati_test2)
sati_n_2 = stats.norm(loc=sati_loc2, scale=sati_scale2)
plt.hist( sati_test2,  
          bins=np.arange(sati_test2.min(), sati_test2.max()+0.2, 0.2), 
          rwidth=0.5 )

x = np.arange(sati_test2.min(), sati_test2.max()+0.2, 0.2)
plt.plot(x, 1100*sati_n_2.pdf(x))
plt.title('全英班平均教學滿意度')
plt.show()
t_21,pv22 =stats.kstest(sati_test2,sati_n_2.cdf)
if pv22 > 0.05:
   print(f'常態分配,P_value : {pv22}')
else:
   print(f'非常態分配,P_value : {pv22}')
# =============================================================================
# t檢定：
# =============================================================================
# #  同質性檢定
# sati_t_f, sati_t_pv = stats.levene(sati_test1, sati_test2)
# print(sati_t_pv)
# 
# sati_tt_f, sati_tt_pv = stats.ttest_ind(sati_test1,sati_test2,equal_var=False)		
# print(sati_tt_pv)		
# 
# # 變異數分析:樣本間是否有顯著差異
# f_value, p_value = stats.f_oneway(sati_test1, sati_test2)
# print(p_value)
# =============================================================================
# 無母數檢定
print(kruskalwallis(sati_test1, sati_test2))
print('=============================================')

# =============================================================================
df_Full_self = pd.read_excel('/Users/wangshuyou/中山招生策略組RA/EMI分析/110-2/110＿中英語課程教學意見＿學習成效比較＿全校.xlsx', sheet_name='org_成效')[['SEM','CRSNO','T_NAME','屬性','全英','mean']]
 
#檢查缺失值
print(df_Full_self.isna().any())
 
self_test1 = df_Full_self[df_Full_self['全英'] == 0]['mean'].values
self_test2 = df_Full_self[df_Full_self['全英'] == 1]['mean'].values


self_table = pd.pivot_table(df_Full_self, values ='mean', index ='屬性', 
                         aggfunc = [len, np.mean , np.std]
                         ) 
print(self_table)
# =============================================================================
# K-W常態性檢定
# 1. 一般班
# =============================================================================
self_loc1, self_scale1 = stats.norm.fit(self_test1)
self_n1 = stats.norm(loc=self_loc1, scale=self_scale1)
plt.hist( self_test1,  
          bins=np.arange(self_test1.min(), self_test1.max()+0.2, 0.2), 
          rwidth=0.5 )
 
x1 = np.arange(self_test1.min(), self_test1.max()+0.2, 0.2)
plt.plot(x1, 11000*self_n1.pdf(x1))
plt.title('一般班平均自評學習成效')
plt.show()

t_21,pv21 = stats.kstest(self_test1,self_n1.cdf)
if pv21 > 0.01:
   print(f'常態分配,P_value : {pv21}')
else:
   print(f'非常態分配,P_value : {pv21}')
 
# 2. 全英班
self_loc2, self_scale2 = stats.norm.fit(self_test2)
self_n2 = stats.norm(loc=self_loc2, scale=self_scale2)
plt.hist( self_test1,  
           bins=np.arange(self_test2.min(), self_test2.max()+0.2, 0.2), 
           rwidth=0.5 )
 
x1 = np.arange(self_test2.min(), self_test2.max()+0.2, 0.2)
plt.plot(x1, 10000*self_n2.pdf(x1))
plt.title('全英班平均自評學習成效')
plt.show()

t_22,pv22 = stats.kstest(self_test2,self_n2.cdf)
if pv22 > 0.05:
   print(f'常態分配,P_value : {pv22}')
else:
   print(f'非常態分配,P_value : {pv22}')


# 無母數檢定
print(kruskalwallis(self_test1, self_test2))

