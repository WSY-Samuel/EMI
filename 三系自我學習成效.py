#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:12:07 2022

@author: wangshuyou
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from scipy.stats.mstats import kruskalwallis

df_self = pd.read_excel('/Users/wangshuyou/中山招生策略組RA/EMI分析/110-2/1102全英及一般班學習成效填答結果.xlsx',sheet_name = 'org_三系')[['SEM','學系','T_NAME','屬性','全英','mean']].sort_values('SEM')

#print(df_self.info())
#print(df_self.isna().any())


# 三系合併
All_table = pd.pivot_table(df_self, values='mean', index='屬性',aggfunc=[len, np.mean, np.std] )
print(All_table)

df_self_All1 = df_self[df_self['全英'] == 0]['mean'].values
df_self_All2 = df_self[df_self['全英'] == 1]['mean'].values


# 1. 三系一般班
# 常態檢定
All_loc, All_scale = stats.norm.fit(df_self_All1)
All_n_1 = stats.norm(loc = All_loc, scale = All_scale)
plt.hist(df_self_All1,bins = np.arange(df_self_All1.min(),df_self_All1.max()+0.2,0.2))

x = np.arange(df_self_All1.min(), df_self_All1.max()+0.2, 0.2)
plt.plot(x, 500*All_n_1.pdf(x))
plt.title('三系一般班自我學習成效')
plt.show()

t_All_1,pv_All_1 = stats.kstest(df_self_All1,All_n_1.cdf)
if pv_All_1 > 0.05:
   print(f'常態分配,P_value : {pv_All_1}')
else:
   print(f'非常態分配,P_value : {pv_All_1}')
   
# 2. 三系全英班
All_loc1,All_scale1 = stats.norm.fit(df_self_All2)
All_n_2 = stats.norm(loc = All_loc1, scale = All_scale1)
plt.hist(df_self_All2, bins = np.arange(df_self_All2.min(), df_self_All2.max()+0.2, 0.2))

x1 = np.arange(df_self_All2.min(), df_self_All2.max()+0.2, 0.2)
plt.plot(x1, 100*All_n_2.pdf(x1))
plt.title('三系全英班自我學習成效')
plt.show()

t_All_2, pv_All_2 = stats.kstest(df_self_All2,All_n_2.cdf)
if pv_All_2 > 0.1:
    print(f'常態分配,P_value : {pv_All_2}')
else:
    print(f'非常態分配,P_value : {pv_All_2}')

print(kruskalwallis(df_self_All1, df_self_All2))

print('============================================')

# 化學
df_self_sci = df_self[df_self['學系'] == '化學']
df_self_sci1 = df_self[(df_self['學系'] == '化學') & (df_self['全英'] == 0)]['mean'].values
df_self_sci2 = df_self[(df_self['學系'] == '化學') & (df_self['全英'] == 1)]['mean'].values

table_sci = pd.pivot_table(df_self_sci,values='mean',index='屬性',aggfunc=[len,np.mean,np.std])
print(table_sci)

# 1. 化學一般班
# 常態檢定
sci_loc, sci_scale = stats.norm.fit(df_self_sci1)
sci_n_1 = stats.norm(loc = sci_loc, scale = sci_scale)
plt.hist(df_self_sci1,bins = np.arange(df_self_sci1.min(),df_self_sci1.max()+0.2,0.2))

x2 = np.arange(df_self_sci1.min(), df_self_sci1.max()+0.2, 0.2)
plt.plot(x2, 100*All_n_1.pdf(x2))
plt.title('化學一般班自我學習成效')
plt.show()

t_sci_1,pv_sci_1 = stats.kstest(df_self_sci1,sci_n_1.cdf)
if pv_sci_1 > 0.1:
   print(f'常態分配,P_value : {pv_sci_1}')
else:
   print(f'非常態分配,P_value : {pv_sci_1}')
   
# 2. 化學全英班
sci_loc1,sci_scale1 = stats.norm.fit(df_self_sci2)
sci_n_2 = stats.norm(loc = sci_loc1, scale = sci_scale1)
plt.hist(df_self_sci2, bins = np.arange(df_self_sci2.min(), df_self_sci2.max()+0.2, 0.2))

x3 = np.arange(df_self_sci2.min(), df_self_sci2.max()+0.2, 0.2)
plt.plot(x3, 20*sci_n_2.pdf(x3))
plt.title('化學全英班自我學習成效')
plt.show()

t_sci_2, pv_sci_2 = stats.kstest(df_self_sci2,sci_n_2.cdf)
if pv_sci_2 > 0.05:
    print(f'常態分配,P_value : {pv_All_2}')
else:
    print(f'非常態分配,P_value : {pv_All_2}')

print(kruskalwallis(df_self_sci1, df_self_sci2))

print('============================================')

# 電機
df_self_ele = df_self[df_self['學系'] == '電機']
df_self_ele1 = df_self[(df_self['學系'] == '電機') & (df_self['全英'] == 0)]['mean'].values
df_self_ele2 = df_self[(df_self['學系'] == '電機') & (df_self['全英'] == 1)]['mean'].values

table_ele = pd.pivot_table(df_self_ele,values='mean',index='屬性',aggfunc=[len,np.mean,np.std])
print(table_ele)

# 1. 電機一般班
# 常態檢定
ele_loc, ele_scale = stats.norm.fit(df_self_ele1)
ele_n_1 = stats.norm(loc = ele_loc, scale = ele_scale)
plt.hist(df_self_ele1,bins = np.arange(df_self_ele1.min(),df_self_ele1.max()+0.2,0.2))

x4 = np.arange(df_self_ele1.min(), df_self_ele1.max()+0.2, 0.2)
plt.plot(x4, 200*ele_n_1.pdf(x4))
plt.title('電機一般班自我學習成效')
plt.show()

t_ele_1,pv_ele_1 = stats.kstest(df_self_ele1,ele_n_1.cdf)
if pv_ele_1 > 0.1:
   print(f'常態分配,P_value : {pv_ele_1}')
else:
   print(f'非常態分配,P_value : {pv_ele_1}')
   
# 2. 電機全英班
ele_loc1,ele_scale1 = stats.norm.fit(df_self_ele2)
ele_n_2 = stats.norm(loc = ele_loc1, scale = ele_scale1)
plt.hist(df_self_ele2, bins = np.arange(df_self_ele2.min(), df_self_ele2.max()+0.2, 0.2))

x5 = np.arange(df_self_ele2.min(), df_self_ele2.max()+0.2, 0.2)
plt.plot(x5, 50*ele_n_2.pdf(x5))
plt.title('電機全英班自我學習成效')
plt.show()

t_ele_2, pv_ele_2 = stats.kstest(df_self_ele2,ele_n_2.cdf)
if pv_ele_2 > 0.1:
    print(f'常態分配,P_value : {pv_ele_2}')
else:
    print(f'非常態分配,P_value : {pv_ele_2}')

print(kruskalwallis(df_self_ele1, df_self_ele2))

print('============================================')

# 機電
df_self_mech = df_self[df_self['學系'] == '機電']
df_self_mech1 = df_self[(df_self['學系'] == '機電') & (df_self['全英'] == 0)]['mean'].values
df_self_mech2 = df_self[(df_self['學系'] == '機電') & (df_self['全英'] == 1)]['mean'].values

table_mech = pd.pivot_table(df_self_mech,values='mean',index='屬性',aggfunc=[len,np.mean,np.std])
print(table_mech)

# 1. 機電一般班
# 常態檢定
mech_loc, mech_scale = stats.norm.fit(df_self_mech1)
mech_n_1 = stats.norm(loc = mech_loc, scale = mech_scale)
plt.hist(df_self_mech1,bins = np.arange(df_self_mech1.min(),df_self_mech1.max()+0.2,0.2))

x6 = np.arange(df_self_mech1.min(), df_self_mech1.max()+0.2, 0.2)
plt.plot(x6, 200*mech_n_1.pdf(x6))
plt.title('機電一般班自我學習成效')
plt.show()

t_mech_1,pv_mech_1 = stats.kstest(df_self_mech1,mech_n_1.cdf)
if pv_mech_1 > 0.1:
   print(f'常態分配,P_value : {pv_mech_1}')
else:
   print(f'非常態分配,P_value : {pv_mech_1}')
   
# 2. 機電全英班
mech_loc1,mech_scale1 = stats.norm.fit(df_self_mech2)
mech_n_2 = stats.norm(loc = mech_loc1, scale = mech_scale1)
plt.hist(df_self_mech2, bins = np.arange(df_self_mech2.min(), df_self_mech2.max()+0.2, 0.2))

x7 = np.arange(df_self_mech2.min(), df_self_mech2.max()+0.2, 0.2)
plt.plot(x7, 50*mech_n_2.pdf(x7))
plt.title('機電全英班自我學習成效')
plt.show()

t_mech_2, pv_mech_2 = stats.kstest(df_self_mech2,mech_n_2.cdf)
if pv_mech_2 > 0.1:
    print(f'常態分配,P_value : {pv_mech_2}')
else:
    print(f'非常態分配,P_value : {pv_mech_2}')

print(kruskalwallis(df_self_mech1, df_self_mech2))
