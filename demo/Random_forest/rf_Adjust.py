# -*- coding:utf-8 -*-

#随机森林调参
#RandomizedSearchCV  随机最佳
#GridSearchCV 地毯式最佳


import pandas as pd
features = pd.read_csv('data/temps_extended.csv')


features = pd.get_dummies(features)

labels = features['actual']
features = features.drop('actual', axis = 1)

feature_list = list(features.columns)

import numpy as np

features = np.array(features)
labels = np.array(labels)

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#################选择6个比较重要的参数当做训练集，重新创建训练集##############################
important_feature_names = ['temp_1', 'average', 'ws_1', 'temp_2', 'friend', 'year']

important_indices = [feature_list.index(feature) for feature in important_feature_names]

important_train_features = train_features[:, important_indices]
important_test_features = test_features[:, important_indices]

print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)

train_features = important_train_features[:]
test_features = important_test_features[:]

feature_list = important_feature_names[:]

#################选择6个比较重要的参数当做训练集，重新创建训练集##############################

########创建随机森林模型###################
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# 打印所有参数
pprint(rf.get_params())

# {'bootstrap': True,#是否随机采样
#  'criterion': 'mse',#指定目标方程   损失的计算方法  熵值 回归   mse计算误差
#  'max_depth': None,# 树的最大深度                     重要
#  'max_features': 'auto',
#  'max_leaf_nodes': None,      最大叶子节点                       重要
#  'min_impurity_decrease': 0.0,
#  'min_impurity_split': None,
#  'min_samples_leaf': 1,          信息增益           重要
#  'min_samples_split': 2,       最小分裂次数           重要
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 'warn',
#  'n_jobs': None,              #多少核CPU 去跑
#  'oob_score': False,
#  'random_state': 42,
#  'verbose': 0,
#  'warm_start': False}

from sklearn.model_selection import RandomizedSearchCV# 随机最好
# 建立树的个数
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# 最大特征的选择方式
max_features = ['auto', 'sqrt']
# 树的最大深度  10 20 none
max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
max_depth.append(None)
# 节点最小分裂所需样本个数
min_samples_split = [2, 5, 10]
# 叶子节点最小样本数，任何分裂不能让其子节点样本数少于此值
min_samples_leaf = [1, 2, 4]
# 样本采样方法
bootstrap = [True, False]

# Random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf = RandomForestRegressor()# 创建模型
#随机寻找参数  cv：交叉验证 ， n_iter 随机100次，scoring：评估方法，verbose：打印信息，n_jobs：所以cpu去跑
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error',
                              cv = 3, verbose=2, random_state=42, n_jobs=-1)




# 执行寻找操作
# rf_random.fit(train_features, train_labels)
# print(rf_random.best_params_)
best_params = {'n_estimators': 1800, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': None, 'bootstrap': True}


def evaluate(model, test_features, test_labels):  #评估
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape

    print('平均气温误差.',np.mean(errors))
    print('Accuracy = {:0.2f}%.'.format(accuracy))



#################使用默认参数##########################
# 平均气温误差. 3.91697080292
# Accuracy = 93.36%.
base_model = RandomForestRegressor( random_state = 42) #使用默认的参数
base_model.fit(train_features, train_labels)
print('默认参数')
evaluate(base_model, test_features, test_labels)
#################使用默认参数##########################


#################使用最好参数##########################
# 平均气温误差. 3.7141472957
# Accuracy = 93.73%.
best_random = RandomForestRegressor(n_estimators=1800,min_samples_split=10,random_state = 42,min_samples_leaf=4,max_features='auto',max_depth=None,bootstrap=True)
best_random.fit(train_features, train_labels)
print('局部最好')
evaluate(best_random, test_features, test_labels)
#################使用最好参数##########################

################在随机最好的参数进行微调######################
# 平均气温误差. 3.69222090145
# Accuracy = 93.77%.
from sklearn.model_selection import GridSearchCV# 地毯式搜索

param_grid = {'n_estimators': [1000, 1200, 1400, 1600],
              'min_samples_split': [3, 5, 7],
              'min_samples_leaf': [2,3, 4, 5,6],
              'max_features': ['auto'],
              'max_depth': [None],
              'bootstrap': [True]}



rf = RandomForestRegressor()

# 网络搜索
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           scoring = 'neg_mean_absolute_error', cv = 3,
                           n_jobs = -1, verbose = 2)
grid_search.fit(train_features, train_labels)
best_grid = grid_search.best_estimator_
evaluate(best_grid, test_features, test_labels)
################在随机最好的参数进行微调######################


########创建随机森林模型###################