# -*- coding:utf-8 -*-
#1.天气最高温预测  随机森林
#2.
import pandas as pd
import datetime
import numpy as np
features = pd.read_csv('data/temps.csv')
# print(features.head())
# print('数据维度：',features.shape)
# print(features.describe())
# print(features.head())
#处理时间数据
years  = features['year']
month  = features['month']
day  = features['day']
dates = [str(int(years)) + '-' + str(int(month)) + '-'+ str(int(day)) for years,month,day in zip(years,month,day)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
import matplotlib.pyplot as plt

###########设置布局 查看数据走势#########################
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))#两行两列 大小10，10
# fig.autofmt_xdate(rotation = 45)
#
# #时间与当天真实最高温 及标签值 预测值
# ax1.plot(dates,features['actual'])
# ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')# 设置XY轴的名称
#
#
#
# # 昨天
# ax2.plot(dates, features['temp_1'])
# ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
#
# # 前天
# ax3.plot(dates, features['temp_2'])
# ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
#
# # 随表标的值
# ax4.plot(dates, features['friend'])
# ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
#
# plt.tight_layout(pad=2)
# plt.show()
###########设置布局 查看数据走势#########################



##############数据预处理########################
##########转化成数值形式########################
# 独热编码
#例：数据中有 Fri,Sat,Sun,Mon,Tues  转化成是否是Fri的形式

features = pd.get_dummies(features)
#提取标签 预测值

labels = np.array(features['actual'])
#在特征中去掉标签
features = features.drop('actual',axis=1)
# 保存特征名称
feature_list = list(features.columns)

#转换成合适的格式
features = np.array(features)




##############数据预处理########################


################训练集与测试集################
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,
                                                                           random_state = 42)
# print('训练集特征:', train_features.shape)
# print('训练集标签:', train_labels.shape)
# print('测试集特征:', test_features.shape)
# print('测试集标签:', test_labels.shape)


################训练集与测试集################


##################建立一个基础的随机森林模型########################
from sklearn.ensemble import RandomForestRegressor

# 建模
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# 训练
rf.fit(train_features, train_labels)

##############预测##########################0.60
# # 预测测试集结果
predictions = rf.predict(test_features)
# #计算误差
# # for i ,j in zip(predictions,test_labels):
# #     print('预测值：',i,'实际值：',j,'误差：',abs(i-j))
#
#
errors = abs(predictions - test_labels)
# #绝对百分比
# mape = 100*(errors/test_labels)
# print(np.mean(mape)) #0.60
print('平均温度误差:', round(np.mean(errors), 2), 'degrees.')#3.83 degrees.

##############预测##########################
##################建立一个基础的随机森林模型########################




# ############可视化展示树###########################
# from sklearn.tree import export_graphviz
# import pydot #pip install pydot
#
# # 拿到其中的一棵树
# tree = rf.estimators_[5]
#
# # 导出成dot文件
# export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
#
# # 绘图
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
#
# # 展示
# # graph.write_png('tree.png');
# ############可视化展示树###########################

############特征重要性##########################

# 得到特征重要性
# importances = list(rf.feature_importances_)
#
# # 转换格式
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#
# # 排序
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#
# # 对应进行打印
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# Variable: temp_1               Importance: 0.7
# Variable: average              Importance: 0.19
# Variable: day                  Importance: 0.03
# Variable: temp_2               Importance: 0.02
# Variable: friend               Importance: 0.02
# Variable: month                Importance: 0.01
# Variable: year                 Importance: 0.0
# Variable: week_Fri             Importance: 0.0
# Variable: week_Mon             Importance: 0.0
# Variable: week_Sat             Importance: 0.0
# Variable: week_Sun             Importance: 0.0
# Variable: week_Thurs           Importance: 0.0
# Variable: week_Tues            Importance: 0.0
# Variable: week_Wed             Importance: 0.0
############特征重要性##########################

############用最重要的特征来训练################### 0.62

# rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
#
# # 拿到这俩特征
# important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
# train_important = train_features[:, important_indices]
# test_important = test_features[:, important_indices]
#
# # 重新训练模型
# rf_most_important.fit(train_important, train_labels)
#
# # 预测结果
# predictions = rf_most_important.predict(test_important)
#
# errors = abs(predictions - test_labels)
#
# # 评估结果
#
# mape = np.mean(100 * (errors / test_labels))
#
# print('mape:', mape)##mape: 6.22905572361  损失变大


############用最重要的特征来训练###################


#############增大数据量#####################
features = pd.read_csv('data/temps_extended.csv')
# print(features.shape)#2191 12
#而且多了新的特征值

years = features['year']
months = features['month']
days = features['day']

# 格式转换
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个季节变量


################pairplot 画图###################
seasons = []

for month in features['month']:
    if month in [1, 2, 12]:
        seasons.append('winter')
    elif month in [3, 4, 5]:
        seasons.append('spring')
    elif month in [6, 7, 8]:
        seasons.append('summer')
    elif month in [9, 10, 11]:
        seasons.append('fall')

# 有了季节我们就可以分析更多东西了
reduced_features = features[['temp_1', 'prcp_1', 'average', 'actual']]

reduced_features['season'] = seasons


import seaborn as sns
sns.set(style="ticks", color_codes=True)

# 选择你喜欢的颜色模板
palette = sns.xkcd_palette(['dark blue', 'dark green', 'gold', 'orange'])

# 绘制pairplot
sns.pairplot(reduced_features, hue = 'season', diag_kind = 'kde', palette= palette, plot_kws=dict(alpha = 0.7),
                   diag_kws=dict(shade=True))# hue 按照春夏秋冬分颜色   plot_kws=dict(alpha = 0.7) 透明程度
# plt.show()  绘制pairplot 图
################pairplot 画图###################

#数据预处理 和上面类似
# 独热编码
features = pd.get_dummies(features)

# 提取特征和标签
labels = features['actual']
features = features.drop('actual', axis = 1)

# 特征名字留着备用
feature_list = list(features.columns)

# 转换成所需格式
import numpy as np

features = np.array(features)
labels = np.array(labels)

# 数据集切分
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size = 0.25, random_state = 0)

original_feature_indices = [feature_list.index(feature) for feature in
                                      feature_list if feature not in
                                      ['ws_1', 'prcp_1', 'snwd_1']]

rf_exp = RandomForestRegressor(n_estimators= 100, random_state=0)
rf_exp.fit(train_features, train_labels)
predictions = rf_exp.predict(test_features)

# 评估
errors = abs(predictions - test_labels)
mape = 100*(errors/test_labels)
print('平均温度误差:', round(np.mean(errors), 2), 'degrees.')#4.05 degrees.
print(np.mean(mape)) #0.66

# 特征名字
importances = list(rf_exp.feature_importances_)

# 名字，数值组合在一起
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# 排序
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# 打印出来
[print('特征: {:20} 重要性: {}'.format(*pair)) for pair in feature_importances];
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]

# 累计重要性
cumulative_importances = np.cumsum(sorted_importances)

important_feature_names = [feature[0] for feature in feature_importances[0:5]]
# 找到它们的名字
important_indices = [feature_list.index(feature) for feature in important_feature_names]

# 重新创建训练集
# important_train_features = train_features[:, important_indices]
# important_test_features = test_features[:, important_indices]
#
# # 数据维度
# print('Important train features shape:', important_train_features.shape)
# print('Important test features shape:', important_test_features.shape)
#
# rf_exp.fit(important_train_features, train_labels);
#
# predictions = rf_exp.predict(important_test_features)
#
# # 评估结果
# errors = abs(predictions - test_labels)
#
# print('平均温度误差:', round(np.mean(errors), 2), 'degrees.')#特征只选比较重要的5个 效果下降 4.11
#
# mape = 100 * (errors / test_labels)
#
# # accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')
#
#
#
#
# #效率
#
# import time
#
# # 这次是用所有特征
# all_features_time = []
#
# # 算一次可能不太准，来10次取个平均
# for _ in range(10):
#     start_time = time.time()
#     rf_exp.fit(train_features, train_labels)
#     all_features_predictions = rf_exp.predict(test_features)
#     end_time = time.time()
#     all_features_time.append(end_time - start_time)
#
# all_features_time = np.mean(all_features_time)
# print('使用所有特征时建模与测试的平均时间消耗:', round(all_features_time, 2), '秒.')#0.76
#
#
# # 这次是用部分重要的特征
# reduced_features_time = []
#
# # 算一次可能不太准，来10次取个平均
# for _ in range(10):
#     start_time = time.time()
#     rf_exp.fit(important_train_features, train_labels)
#     reduced_features_predictions = rf_exp.predict(important_test_features)
#     end_time = time.time()
#     reduced_features_time.append(end_time - start_time)
#
# reduced_features_time = np.mean(reduced_features_time)
# print('使用所有特征时建模与测试的平均时间消耗:', round(reduced_features_time, 2), '秒.')#0.46

#############增大数据量#####################
