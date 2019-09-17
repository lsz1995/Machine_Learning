# -*- coding:utf-8 -*-
import pandas as pd

df = pd.read_csv('../data/titanic.csv')
#默认读5条
print(df.head())
#返回当前信息
print(df.info())
#索引
print(df.index)
#列名
print(df.columns)
#数值
print(df.values)

#取指定数据 series 一列
print(df['Age'][2:5])

print(df['Age'].unique() )# 取唯一值