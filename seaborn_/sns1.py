# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

np.random.seed(sum(map(ord,'distributions')))


# x = np.random.normal(size=100)#高斯分布的数据
# # sns.distplot(x,kde=False)#直方图   会根据数据的范围自动设置
# sns.distplot(x,bins=20,kde=False)# x 轴 分成20


# x = np.random.gamma(6, size=200)
# sns.distplot(x, kde=False, fit=stats.gamma)# fit 统计指标
# plt.show()


#点变量 根据均值和协方差生成数据

# mean , cov =[0,1],[(1,.5),(.5,1)]
# data = np.random.multivariate_normal(mean,cov,200)#用于根据实际情况生成一个多元正态分布矩阵
# df = pd.DataFrame(data,columns=['x','y'])


# #观察两个变量之间的分布关系 散点图
# sns.jointplot(x='x',y='y',data=df)
# sns.jointplot(x='x',y='y',kind='hex',data=df)#hex 图 数据量大的化 得到数据多的区域
# plt.show()


#两两特征直接关系
# iris = sns.load_dataset('iris') #
# sns.pairplot(iris) #pairplot 两两特征直接的关系
# plt.show()



#回归分析绘图

# tips = sns.load_dataset('tips')# 内置数据集
# print(tips.head(5))
# #regplot  mplot  都可以绘制回归
#
# # sns.regplot(x='total_bill',y='tip',data=tips)#消费金额与小费的关系
# sns.regplot(x='size',y='tip',data=tips,x_jitter=.05)#人数与小费的关系  x_jitter:加浮动
#
# plt.show()

#回归分析绘图


#多变量分析绘图

tips = sns.load_dataset('tips')# 内置数据集
titanic = sns.load_dataset('titanic')
iris = sns.load_dataset('iris') #


# sns.stripplot(x='day',y='total_bill',data=tips,jitter=True) #1
# sns.swarmplot(x='day',y='total_bill',data=tips,hue='sex') #  hue ： 在哪个特征进行划分2
# sns.boxplot(x='day',y='total_bill',data=tips,hue='time')#3 盒图  离群点
# sns.violinplot(x='day',y='total_bill',data=tips,hue='time',split=True)#3 小提琴图  离群点
# plt.show()

#多变量分析绘图

#分类属性绘图
# sns.barplot(x='sex',y='survived',hue='class',data=titanic)# 性别与存活 按照船舱分类
# sns.pointplot(x='sex',y='survived',hue='class',data=titanic)#点图

# sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker')#factorplot 默认折线图
# sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker',kind='bar')#factorplot bar 条形图
# sns.factorplot(x='day',y='total_bill',data=tips,hue='smoker',kind='swarm',col='time')#factorplot swarm dian
# plt.show()
# * x,y,hue 数据集变量 变量名
# * date 数据集 数据集名
# * row,col 更多分类变量进行平铺显示 变量名
# * col_wrap 每行的最高平铺数 整数
# * estimator 在每个分类中进行矢量到标量的映射 矢量
# * ci 置信区间 浮点数或None
# * n_boot 计算置信区间时使用的引导迭代次数 整数
# * units 采样单元的标识符，用于执行多级引导和重复测量设计 数据变量或向量数据
# * order, hue_order 对应排序列表 字符串列表
# * row_order, col_order 对应排序列表 字符串列表
# * kind : 可选：point 默认, bar 柱形图, count 频次, box 箱体, violin 提琴, strip 散点，swarm 分散点
# size 每个面的高度（英寸） 标量
# aspect 纵横比 标量
# orient 方向 "v"/"h"
# color 颜色 matplotlib颜色
# palette 调色板 seaborn颜色色板或字典
# legend hue的信息面板 True/False
# legend_out 是否扩展图形，并将信息框绘制在中心右边 True/False
# share{x,y} 共享轴线 True/False
#分类属性绘图


#
# g = sns.FacetGrid(tips,col='time')# 实例化
# g.map(plt.hist,'tip')# 指定好  条形图
# plt.show()



# g = sns.FacetGrid(tips,col='sex',hue='smoker') #性别与是否吸烟
# g.map(plt.scatter,'total_bill','tip',alpha=.7) #总花费与小费的关系
# g.add_legend()#添加类别 smoker


# g = sns.FacetGrid(tips,col='sex',row='smoker',margin_titles=True) #
# g.map(sns.regplot,'size','total_bill',color='.1',fit_reg =False,x_jitter=.1)


# pal = dict(Lunch="seagreen", Dinner="gray")#指定颜色
# g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
# g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
# g.add_legend()

# g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})#指定形状
# g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
# g.add_legend()

# with sns.axes_style("white"):
#     g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
# g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5);
# g.set_axis_labels("Total bill (US Dollars)", "Tip")# X Y轴名字
# g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])#x y 轴取值
# g.fig.subplots_adjust(wspace=.02, hspace=.02)#xy 轴范围
#g.fig.subplots_adjust(left  = 0.125,right = 0.5,bottom = 0.1,top = 0.9, wspace=.02, hspace=.02)


# iris = sns.load_dataset("iris")
# g = sns.PairGrid(iris)
# g.map(plt.scatter);
#

#heatmap  热力图

# uniform_data = np.random.rand(3, 3)
# print (uniform_data)
# heatmap = sns.heatmap(uniform_data)



# flights = sns.load_dataset("flights")
# flights = flights.pivot( "year","month", "passengers")
#
# print (flights)
# ax = sns.heatmap(flights,annot=True,fmt='d')


plt.show()