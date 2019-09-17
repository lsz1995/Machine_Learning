# -*- coding:utf-8 -*-
#房价预测   逻辑回归
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler #标准化数据
from scipy import stats
import warnings
warnings.filterwarnings('ignore')#去掉警告

df_train =pd.read_csv('../data/fj_data/train.csv')
# print(df_train.head())#打印5条

# print(df_train['SalePrice'].describe())#查看价格（标签）属性
# count      1460.000000
# mean     180921.195890
# std       79442.502883
# min       34900.000000
# 25%      129975.000000
# 50%      163000.000000
# 75%      214000.000000
# max      755000.000000
# Name: SalePrice, dtype: float64

#1.查看是否满足正态分布

# f,ax = plt.subplots(1,1,figsize=(15,8))#初始化一行两列的子图   figsize 长宽
# sns.distplot(df_train['SalePrice'])
# plt.show()

print('偏度值:{}'.format(df_train['SalePrice'].skew()))
print('峰度值:{}'.format(df_train['SalePrice'].kurt()))

# 面积与售卖价格的关系
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #面积与售卖价格的关系
# data.plot.scatter(x=var,y='SalePrice',ylim =(0,800000)) #散点图
# plt.show()

# 地下室与售卖价格的关系
# var = 'TotalBsmtSF'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #地下室面积与售卖价格的关系
# data.plot.scatter(x=var,y='SalePrice',ylim =(0,800000)) #散点图
# plt.show()

# 整体材料和饰面质量与售卖价格的关系
# var = 'OverallQual'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #地下室面积与售卖价格的关系
# data.plot.scatter(x=var,y='SalePrice',ylim =(0,800000)) #散点图
# plt.show()

# 原施工日期与售卖价格的关系
# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #地下室面积与售卖价格的关系
# data.plot.scatter(x=var,y='SalePrice',ylim =(0,800000)) #散点图
# plt.show()


#特征相关性

corrmat = df_train.corr()
# f, ax =plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,square=True,cmap='BrBG_r')#所有特征的相关性
# plt.show()
k=10
cols =corrmat.nlargest(k,'SalePrice')['SalePrice'].index#选十个和出售价格的相关系数最高的10个特征的索引
cm = np.corrcoef(df_train[cols].values.T)#
sns.set(font_scale=1.25)
# hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols)
# plt.show()

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size = 2.5)
# plt.show();


#观察缺失值

total = df_train.isnull().sum().sort_values(ascending=False)
percen = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)#当前特征缺失值占总数的比例
missing_data = pd.concat([total,percen],axis=1,keys=['total','percent'])


# print(missing_data.head(20))
#               total   percent
# PoolQC         1453  0.995205
# MiscFeature    1406  0.963014
# Alley          1369  0.937671
# Fence          1179  0.807534
# FireplaceQu     690  0.472603
# LotFrontage     259  0.177397
# GarageCond       81  0.055479
# GarageType       81  0.055479
# GarageYrBlt      81  0.055479
# GarageFinish     81  0.055479
# GarageQual       81  0.055479
# BsmtExposure     38  0.026027
# BsmtFinType2     38  0.026027
# BsmtFinType1     37  0.025342
# BsmtCond         37  0.025342
# BsmtQual         37  0.025342
# MasVnrArea        8  0.005479
# MasVnrType        8  0.005479
# Electrical        1  0.000685
# Utilities         0  0.000000


#读取训练集和测试集
train = pd.read_csv('../data/fj_data/train.csv')
test = pd.read_csv('../data/fj_data/test.csv')

#数据大小
# print(train.shape)
# print(test.shape)

#ID
train_ID = train['Id']
test_ID = test['Id']

#去掉ID
train.drop('Id',axis=1,inplace=True)#在原数据生效
test.drop('Id',axis=1,inplace=True)#在原数据生效
#

#解决离群点
#面积中的利群点参考上面的面积与售卖价格的关系
train = train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<300000)].index)#删除面积大于4000 且价格小于300000的数据 不正常的点
# var = 'GrLivArea'
# data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1) #面积与售卖价格的关系
# train.plot.scatter(x=var,y='SalePrice',ylim =(0,800000)) #散点图
# plt.show()


##################样本正态分布变换  对数函数

# sns.distplot(train['SalePrice'] , fit=norm);
#
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #分布图
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #QQ图
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()


train['SalePrice'] = np.log1p(train['SalePrice'])#对数变换

# sns.distplot(train['SalePrice'] , fit=norm);
#
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #分布图
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #QQ图
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()


#缺失值处理

ntrain = train.shape[0]#训练集数
ntest = test.shape[0]#测试集数
y_train = train.SalePrice.values#训练集结果集
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
# print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100 #各个特征缺失值百分比

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]#取缺失值最多的前30个

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})#构造成 一个dataframe
# print('前20：',missing_data.head(20))

print(all_data["PoolQC"][:5])#是否有游泳池
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")#填充空值
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))#填充中位值

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data = all_data.drop(['Utilities'], axis=1)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape
print('Shape all_data: {}'.format(all_data.shape))

#增加一个新特征总面积
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


from scipy.stats import norm, skew

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)#计算偏度值 ，排序
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))#偏多太大 不满足正态分布


#Box-Cox
#1. 找到偏度大于0.75的
skewness = skewness[abs(skewness)>0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))


from scipy.special import boxcox1p    #Box-Cox变换
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)


train = all_data[:ntrain]
test = all_data[ntrain:]


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# import xgboost as xgb
n_folds = 5#交叉验证倍数

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


#预处理  去除离群点
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))#预处理离群点，

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

# model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
#                              learning_rate=0.05, max_depth=3,
#                              min_child_weight=1.7817, n_estimators=2200,
#                              reg_alpha=0.4640, reg_lambda=0.8571,
#                              subsample=0.5213, silent=1,
#                              nthread = -1)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = rmsle_cv(model_xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))



