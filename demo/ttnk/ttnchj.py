# -*- coding:utf-8 -*-
#逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')#plt 风格
import warnings
warnings.filterwarnings('ignore')#去掉警告
data = pd.read_csv('../data/titanic.csv')#读取数据




# print(data.isnull().sum())#  查看确定缺失值
# print(data.describe())#过滤掉不是数据型的 统计信息
#
#---------------------------------------#
# f,ax = plt.subplots(1,2,figsize=(15,8))#初始化一行两列的子图   figsize 长宽
# data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)#画饼图 Survived是否获救 0 1
#
# ax[0].set_title('Survived')
# ax[0].set_ylabel('')
# sns.countplot('Survived',data=data,ax=ax[1])#柱状图
# ax[1].set_title('Survived')
# plt.show()

#---------------------------------------#

# print(data.groupby(['Sex','Survived'])['Survived'].count())  #年龄 获救的统计  女性获救几率大于男性

# Sex     Survived
# female  0            81
#         1           233
# male    0           468
#         1           109
# Name: Survived, dtype: int64

#---------------------------------------#
# print(pd.crosstab(data.Pclass,data.Survived,margins=True))# 等级仓与获救关系
# Survived    0    1  All
# Pclass
# 1          80  136  216
# 2          97   87  184
# 3         372  119  491
# All       549  342  891
#---------------------------------------#
# print(pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True))#等级仓性别  与获救关系  有关
# Pclass             1    2    3  All
# Sex    Survived
# female 0           3    6   72   81
#        1          91   70   72  233
# male   0          77   91  300  468
#        1          45   17   47  109
# All              216  184  491  891

# print(data['Age'].max())
# print(data['Age'].min())
# print(data['Age'].mean())
#可把年龄分组 填充值

data['Initial']=0
for i in data:
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33#判断更改
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46
# data.loc[(data.Age.isnull())&(data.Initial=='Capt'),'Age']=46
#---------------------------------------#
# print('登入地点与存活')
# print(pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True))
#
# Sex             female      male       All
# Survived             0    1    0    1
# Embarked Pclass
# C        1           1   42   25   17   85
#          2           0    7    8    2   17
#          3           8   15   33   10   66
# Q        1           0    1    1    0    2
#          2           0    2    1    0    3
#          3           9   24   36    3   72
# S        1           2   46   51   28  127
#          2           6   61   82   15  164
#          3          55   33  231   34  353
# All                 81  231  468  109  889
#c港生存率高
# sns.factorplot('Embarked','Survived',data=data)
# fig = plt.gcf()
# fig.set_size_inches(5,3)
# plt.show()
#---------------------------------------#
#特征的相关性

# sns.heatmap(data.corr(),annot=True,cmap='RdYlGn_r',linewidths=0.2) #各个特征相关性
# fig =plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()

data['Embarked'].fillna('S',inplace=True)#直接修改原对象 缺失值填充
#-----------特征工程----------------------------#
#年龄转化成数字区间
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
# print(data.head(2))
#家庭人数转化成数字区间
data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')#check
data['Family_Size']=0
data['Family_Size']=data['Parch']+data['SibSp']#family size
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1#Alone


#船票价格切分
data['Fare_Range']=pd.qcut(data['Fare'],4)
data.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')

data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

#字符串转化成数值
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap ='RdYlGn_r',linewidths=0.2,annot_kws={'size':20})
# fig =plt.gcf()
# fig.set_size_inches(18,15)
# plt.xticks(fontsize =14)
# plt.yticks(fontsize =14)
# plt.show()
# print(data.isnull)

#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]#训练集X
train_Y=train[train.columns[:1]]#训练集X的结果 标签
test_X=test[test.columns[1:]]#测试集
test_Y=test[test.columns[:1]]#测试集结果
X=data[data.columns[1:]]
Y=data['Survived']

model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)#模型的预测值
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))#模型的预测值与真正的数值的对比



model1=svm.SVC(kernel='linear',C=1,gamma=0.1)
model1.fit(train_X,train_Y)
prediction2 = model1.predict(test_X)
print('Accuracy for linear SVM is ',metrics.accuracy_score(prediction2,test_Y))


model  = LogisticRegression() #决策树
model.fit(train_X,train_Y)
prediction3 = model.predict(test_X)
print('Accuracy for LogisticRegression is ',metrics.accuracy_score(prediction3,test_Y))#模型评估metrics.accuracy_score


a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X,train_Y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_Y)))
# plt.plot(a_index, a)
# plt.xticks(x)
# fig=plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()
print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())


#交叉验证 各个算法
from sklearn.model_selection import KFold #for K-fold cross validation分为几个
from sklearn.model_selection import cross_val_score #score evaluation预测精度
from sklearn.model_selection import cross_val_predict #prediction预测结果
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts 分成10折 ，
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:#循环每个算法
    model = i
    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())#标准差
    accuracy.append(cv_result)
new_models_dataframe2= pd.DataFrame({'CV Mean':xyz,'Std':std},index = classifiers)
print(new_models_dataframe2)
#                       CV Mean       Std
# Linear Svm           0.793471  0.047797
# Radial Svm           0.828290  0.034427
# Logistic Regression  0.805843  0.021861
# KNN                  0.813783  0.041210
# Decision Tree        0.810350  0.026123
# Naive Bayes          0.801386  0.028999
# Random Forest        0.814844  0.032465

#寻找最佳参数
from sklearn.model_selection import GridSearchCV # 指定一系列参数值
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_) # 支持向量积最好的结果
print(gd.best_estimator_)#最好的结果的参数

#集成

from sklearn.ensemble import VotingClassifier#投票
ensemble_lin_rbf = VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ],
                       voting='soft').fit(train_X,train_Y)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())

#并行100个
from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',result.mean())



model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,test_Y))
result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
print('The cross validated score for bagged Decision Tree is:',result.mean())

#标准化：  x-均值 除以标准差
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(data[['','']]) #那一列
df_std = std_scale.transform(data[['','']])#转化

#归一化： x-x最小值 / x最大值 - x最小值

min_scale = preprocessing.MinMaxScaler().fit(data[['','']]) #那一列
df_minmax = min_scale.transform(data[['','']])#转化