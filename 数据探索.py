# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:51:13 2019
说明：这份数据集是金融数据（非原始数据，已经处理过了），我们要做的是预测贷款用户是否会逾期。表格中 "status" 是结果标签：0表示未逾期，1表示逾期。
数据集涉密，不要开源到网上，谢谢~
要求：数据切分方式 - 三七分，其中测试集30%，训练集70%，随机种子设置为2018

任务1：对数据进行探索和分析。时间：2天
数据类型的分析
无关特征删除
数据类型转换
缺失值处理




@author: asus
"""
#%%
import pandas as pd
import seaborn as sns
#%% 读取数据
data = pd.read_excel("data.xlsx")
#%% 打印数据信息
print (data.info())
#%% 缺省值比较多,看看缺失较多的属性情况
print (data.isnull().sum().sort_values(ascending = False))
# %%=============================================================================
#    一共4754行数据,其中有部分空值,大部分都有缺失,考虑到大部分属性意义不明,
#暂时不丢弃,也不考虑直接填充。先考虑去除一些明显无用的列。
# =============================================================================
#查看对每一种属性包含的值
# =============================================================================
# n=1
# for name in list(data.columns):
#     print ("#" + str(n) + "-" + name +":")
#     n = n + 1
#     print (data[name].unique())
#     a = input()
# =============================================================================

#%% trade_no,bank_card_no,source,id_name是无效的数据可以剔除
D_data = data
D_data.drop(['trade_no','bank_card_no','source','id_name'],inplace = True,axis = 1)
#%%
print (D_data.info())
#%%
# student_feature  空值填为0
D_data['student_feature'].fillna(0,inplace = True)
#%% 部分缺失值用0填充
D_data['rank_trad_1_month'].fillna(0,inplace = True)
D_data['avg_consume_less_12_valid_month'].fillna(0,inplace = True)
D_data['low_volume_percent'].fillna(0,inplace = True)
D_data['middle_volume_percent'].fillna(0,inplace = True)
D_data['loans_latest_day'].fillna(0,inplace = True)
D_data['latest_query_day'].fillna(0,inplace = True)
#%%
# reg_preference_for_trad 处理空值为"其他城市"
#需要使用独热编码
D_data['loans_latest_day'].fillna("其他城市",inplace = True)
D_data.drop(['loans_latest_day'],axis = 1,inplace = True)
#%% 随机森林
y = D_data['status']
x = D_data[D_data.loc[:,D_data.columns != 'status'].columns]
#%%
from sklearn.ensemble import RandomForestClassifier
#%%
forest_clf = RandomForestClassifier()
forest_clf.fit(x,y)
# 随即森林自带“feature_importances_”属性，可以计算出每个特征的重要性。
features_weights = dict(zip(D_data.columns,forest_clf.feature_importances_))
fearures_weights_sorted = sorted(features_weights.items(),key = lambda x:x[1],
                                 reverse=True)
#%% 降序排列，选择前15个
for item in range(15):
    print(features_weights_sorted[item][0],':',features_weights_sorted[item][1])






