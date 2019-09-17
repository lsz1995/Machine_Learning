# -*- coding:utf-8 -*-
#北京3000条房天下 二手房数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler #标准化数据
from scipy import stats
import warnings
warnings.filterwarnings('ignore')#去掉警告


df_train = pd.read_csv('./fj_data/ftx.csv')
df_train.drop('id',axis=1,inplace=True)#在原数据生效 去掉ID
df_train.drop('title',axis=1,inplace=True)#在原数据生效 去掉名称
df_train.drop('city',axis=1,inplace=True)#都是北京 所以去掉
df_train.drop('price',axis=1,inplace=True)#有每平米价格 所以去掉总价
df_train.replace('暂无',np.nan,inplace=True)# 把所有‘暂无替换为缺失值
df_train.loc[df_train.longitude==0,'longitude']=np.nan#判断更改  如果经度为0 则为缺失值
df_train.loc[df_train.latitude==0,'latitude']=np.nan#判断更改  如果纬度为0 则为缺失值

df_train['room_type']=df_train['room_type'].str.replace('室','_').str.replace('厅','_').str.replace('卫','_')#把室厅卫拆成三个数据

df_train['room']=df_train['room_type'].str.split('_',expand=True)[0]
df_train['hall']=df_train['room_type'].str.split('_',expand=True)[1]
df_train['toilet']=df_train['room_type'].str.split('_',expand=True)[2]
df_train.drop('room_type',axis=1,inplace=True)

# var = 'area'
# data = pd.concat([df_train['unit_price'],df_train[var]],axis=1) #面积与售卖价格的关系
# data.plot.scatter(x=var,y='unit_price') #散点图
# plt.show()


df_train['longitude'] = df_train['longitude'].fillna(df_train['longitude'].mean())#用经纬度的平均值填充缺失值
df_train['latitude'] = df_train['latitude'].fillna(df_train['latitude'].mean())

df_train['oriented']=df_train['oriented'].replace(['东','南','西','北','东北','东南','东西','南北','西北','西南'],[1,2,3,4,5,6,7,8,9,10])
# print(df_train['oriented'].mode())#朝向的众数8
df_train.loc[df_train.oriented.isnull(),'oriented']=8#如果朝向为空  填充众数
# print(df_train['decoration'].mode())#装修的众数3
df_train['decoration']=df_train['decoration'].replace(['豪华装修','简装修','精装修','毛坯','中装修',],[1,2,3,4,5])
df_train.loc[df_train.decoration.isnull(),'decoration']=3

df_train['area']=df_train['area'].replace(['北京周边','昌平','朝阳','大兴','东城','房山','丰台','海淀','怀柔','门头沟','密云','平谷','石景山','顺义','通州','西城','延庆','燕郊'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])#地区处理成数值


# print(df_train['room'].mode())#室的众数2
df_train.loc[df_train.room.isnull(),'room']=2
# print(df_train['hall'].mode())#厅的众数2
df_train.loc[df_train.hall.isnull(),'hall']=2
# print(df_train['toilet'].mode())#卫的众数1
df_train.loc[df_train.toilet.isnull(),'toilet']=1


df_train['unit_price']=df_train['unit_price'].str.replace('元/平米','').astype(float)#把室厅卫拆成三个数据
df_train['size']=df_train['size'].str.replace('平米','').astype(float)#把室厅卫拆成三个数据

# df_train.loc[df_train['floor'].filter(regex='低'),'floor']='低'#如果朝向为空  填充众数
# print(df_train.isnull().sum())#统计缺失值   处理完成
df_train.loc[df_train['floor'].str.contains('低'),'floor']='低'
# print(df_train.isnull().sum())#统计缺失值   处理完成
df_train.loc[df_train['floor'].str.contains('中'),'floor']='中'
df_train.loc[df_train['floor'].str.contains('高'),'floor']='高'

df_train.loc[df_train['floor']=='低','floor']=1
df_train.loc[df_train['floor']=='中','floor']=2
df_train.loc[df_train['floor']=='高','floor']=3
df_train.loc[df_train['elevator']=='无','elevator']= 0
df_train.loc[df_train['elevator']=='有','elevator']= 1
df_train['age']=df_train['age'].str.replace('年','')#把室厅卫拆成三个数据
#数据清洗——————————————————————————————————————————————————————————



df_train.drop('longitude',axis=1,inplace=True)#在原数据生效 去掉ID
df_train.drop('latitude',axis=1,inplace=True)#在原数据生效 去掉ID
print(df_train.head(1)['unit_price'])
# print(df_train.head(1))

df_train = df_train.drop(df_train[(df_train['unit_price']>150000)|(df_train['unit_price']<4000)].index)#删除面积大于4000 且价格小于300000的数据 不正常的点
print(df_train['unit_price'].describe())#  最大23w   大于15W当做异常值 去除
# #分布图
# f,ax = plt.subplots(1,1,figsize=(15,8))#初始化一行两列的子图   figsize 长宽
# sns.distplot(df_train['unit_price'])
# plt.show()


from scipy.stats import norm, skew

numeric_feats = df_train.dtypes[df_train.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)#计算偏度值 ，排序
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

#Box-Cox
#1. 找到偏度大于0.75的
skewness = skewness[abs(skewness)>0.75]
# print(skewness)
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
from scipy.special import boxcox1p    #Box-Cox变换
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df_train[feat] = boxcox1p(df_train[feat], lam)


# #########对不满足正态分布的进行对数变换
from sklearn.model_selection import train_test_split #training and testing data split
print(df_train.head(1)['unit_price'])

#查看价格分布
# f,ax = plt.subplots(1,1,figsize=(15,8))#初始化一行两列的子图   figsize 长宽
# sns.distplot(df_train['unit_price'])
# plt.show()
##################


# #相关系数
#
# sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn_r',linewidths=0.2) #各个特征相关性
# fig =plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()
# #相关系数
all_data = pd.get_dummies(df_train)# 所有数据#把数据转换成0 1    如area有0 1 2   转变成area1   area2  area0  结果是0 1





# all_data = df_train# 所有数据

# print(all_data.head())
train = all_data[:2900]
print(train.head(1)['unit_price'])
test = all_data[2900:]
train_Y =train['unit_price']
test_Y = train['unit_price']
train.drop('unit_price',axis=1,inplace=True)
test.drop('unit_price',axis=1,inplace=True)


# print(train.head(1),train_Y.head(1))
##########分训练集和测试集


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC,LassoCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

#模型评估
n_folds = 5#交叉验证倍数
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, train_Y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#make_pipeline  数据预处理  处理掉离群点

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

score = rmsle_cv(lasso)
#算损失
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
# #平均
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
#
# #损失  Lasso  最低
# # Lasso score: 0.0007 (0.0000)
# #
# # ElasticNet score: 0.0009 (0.0000)
# #
# # Kernel Ridge score: 0.0270 (0.0029)
# #
# # Gradient Boosting score: 0.3495 (0.0433)
# #
# #  Averaged base models score: 0.0887 (0.0118)
#
#
# model=lasso
# model.fit(train,y_train)
# prediction1=model.predict(test)#模型的预测值
# print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,y_test))#模型的预测值与真正的数值的对比




GBoost.fit(train,train_Y)
prediction1=GBoost.predict(test)#模型的预测值


for i,j in zip(prediction1,test_Y):
    print('预测值：',i,'实际值：',j,'相差:',abs(i-j)/j)


