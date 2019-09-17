import numpy as np

a = [1, 2, 3, 4]
b = np.array(a)         # array([1, 2, 3, 4])
print(b)                # [1 2 3 4]
print(type(b))          # <type 'numpy.ndarray'>
print(b.shape)          # (4,)
print(b.size)           # 4
print(b.argmax())       # 3
print(b.max())          # 4
print(b.mean())         # 2.5

print("---------------------")

c = [[1, 2], [3, 4]]    # 二维列表
d = np.array(c)         # 二维numpy数组
print(d)
"""
[[1 2]
 [3 4]]
"""
print(d.shape)          # (2, 2)
print(d.size)           # 4
print(d.max(axis=0))    # [3 4]          # 每一列的最大值
print(d.min(axis=1))    # [1 3]          # 每一行的最小值
print(d.mean())         # 2.5            # 平均数
print(d.mean(axis=0))   # [2. 3.]        # 每一列的平均数
print(d.flatten())      # [1 2 3 4]      # 展开一个numpy数组为1维数组
print(np.ravel(c))      # [1 2 3 4]      # 展开一个可以解析的结构为1维数组

print("---------------------")

# 3x3的浮点型2维数组，并且初始化所有元素值为1
e = np.ones((3, 3), dtype=np.float)
print(e)
"""
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
"""
print(e.shape)      # (3, 3)
print(e.size)       # 9

print("---------------------")

# 创建一个一维数组，元素值是把3重复4次，array([3, 3, 3, 3])
f = np.repeat(3, 4)
print(f)            # [3 3 3 3]

print("---------------------")

# 2x2x3的无符号8位整型3维数组，并且初始化所有元素值为0
g = np.zeros((2, 2, 3), dtype=np.uint8)
print(g)
"""
[[[0 0 0]
  [0 0 0]]
 [[0 0 0]
  [0 0 0]]]
"""
print(g.shape)      # (2, 2, 3)

# 用另一种类型表示
h = g.astype(np.float)
print(h)
"""
[[[0. 0. 0.]
  [0. 0. 0.]]
 [[0. 0. 0.]
  [0. 0. 0.]]]
"""

print("---------------------")

# 类似range，array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
l = np.arange(10)
print(l)        # [0 1 2 3 4 5 6 7 8 9]

# 等差数列，0到6之间5个取值，array([ 0., 1.5, 3., 4.5, 6.])
m = np.linspace(0, 6, 5)
print(m)        # [0.  1.5 3.  4.5 6. ]

print("---------------------")

p = np.array(
    [[1, 2, 3, 4],
     [5, 6, 7, 8]]
)
print(p)
"""
[[1 2 3 4]
 [5 6 7 8]]
"""

np.save('p.npy', p)     # 保存到文件
q = np.load('p.npy')    # 从文件读取
print(q)
"""
[[1 2 3 4]
 [5 6 7 8]]
"""


