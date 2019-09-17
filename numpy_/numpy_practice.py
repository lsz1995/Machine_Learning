# -*- coding:utf-8 -*-
import numpy as np

print(np.__version__)
#全零矩阵
z = np.zeros((5,5))
print(z)

#打印一个函数的帮助文档
print(help(np.info(np.add)))

#创建一个10-49数组，倒序排列
a = np.arange(10,50,1)
a = a[::-1]
print(a)
#找数组中不为0的索引
print(np.nonzero([1,2,3,4,5,0,0,123,0,1]))

#随机构造一个3*3矩阵 并打印最大与最小
b = np.random.random((3,3))
print(b)
print(b.min())
print(b.max())
#构造一个5*5矩阵，都为1 ，在最外层加上一圈0
c = np.ones((5,5))
print(np.pad(c,pad_width= 1,constant_values=0,mode='constant' ))#pad_width=1  一圈  constant_values 加什么

#构建一个（6，7,8）的 矩阵 并找到第100个函数

print(np.unravel_index(100,(6,7,8)))

#对5*5矩阵做归一化操作, 归一化： 数组中每个值减去最小值 / 最大值 -最小值
d = np.random.random((5,5))
d_max = d.max()
d_min = d.min()
d = (d-d_min)/(d_max-d_min)
print('对5*5矩阵做归一化操作',d)


#找到两个数组相同的值
e1 = np.random.randint(0,10,10)
e2 = np.random.randint(0,10,10)

print(np.intersect1d(e1,e2))


#得到今天 明天 昨天的日期

today = np.datetime64('today','D')#
yesterday = np.datetime64('today','D') -np.timedelta64(1,'D')
tomomorow = np.datetime64('today','D') +np.timedelta64(1,'D')

#得到一个月中的所有天
f = np.arange('2017-05','2017-06',dtype = 'datetime64[D]')
print(f)

#得到一个数的整数部分
g = np.random.uniform(0,10,10)
print(np.floor(z))


#在一个数组中找到最接近一个数的索引
h = np.arange(100)
i = np.random.uniform(0,100)
index = (np.abs(h-i)).argmin()
print(index)

#32float 和32 int 转换
j = np.arange(10,dtype=np.int32)
j = j.astype(np.float32)

#打印数组元素位置与数值
l = np.arange(9).reshape(3,3)
for index,value in np.ndenumerate(l):
    print(index, value)
#按照数组的某一列进行排序
m = np.random.randint(0,10,(3,3))
print(m[m[:,1].argsort()])

#统计数组中的每一个数值的出现次数
o = np.array([1,1,1,2,2,3,3,4,5,8,9])
#[0 3 2 2 1 1 0 0 1]
print(np.bincount(o))

#对一个思维数组的最后两维求和
p = np.random.randint(0,10,(4,4,4,4))
print(p.sum(axis=(-2,-1)))

#交换矩阵中的两行
q = np.arange(25).reshape(5,5)
print(q)
q[[0,1]] = q[[1,0]]
print(q)

#找一个数组中最常出现的数组

r = np.random.randint(0,10,50)
print(r)
print(np.bincount(r).argmax())

#快速查找top k
r = np.arange(10000)
np.random.shuffle(r)
n = 5
print(r[np.argpartition(-r,n)[:n]])

