import numpy as np

# ndim 维度 shape 行数列数 size 元素个数
arrays = np.array([[1, 2, 3], [2, 3, 4]])
print(arrays)

print('dimension ', arrays.ndim)
print('shape ', arrays.shape)
print('size ', arrays.size)

# 数据的创建部分

# empty为接近0的数
print(np.empty((3, 4)))
# arange负责连续数组（类似于range）
print(np.arange(2, 10, 2))
# linspace则负责进行范围内的随机数据生成，当然，也是顺序的
print(np.linspace(2, 10, 15))

# 数据形状的改变， reshape
print(arrays.reshape((1, 6)))

# np的几种基本的计算方式(1)

# 矩阵的点乘直接乘可还行，便于卷积运算
a = np.arange(2, 10, 2).reshape((2, 2))
b = np.arange(8, 24, 4).reshape((2, 2))
print(a, '\n', b)
print(a * b)

# 标准的矩阵乘法计算
print(np.dot(a,b))

# 列统计有点意思，设置axis为0时即可
print(np.sum(a, axis=0))


# np的几种基本的计算方式(2)
test = np.arange(2,14).reshape((3,4))

# 获取矩阵(整体)最大最小元素的索引argmax，argmin
print(np.argmax(test),np.argmin(test))

# 求取前缀和的函数 cumsum
print(np.cumsum(test))

# 裁剪函数clip，负责进行范围修正
print(np.clip(test,5,9))


# 矩阵的合并操作
a = np.ones((1,3))
b = np.ones((1,3))*4

# 上下合并 vertical stack
print(np.vstack((a,b)))

# 水平合并 horizontal stack
print(np.hstack((a,b)))

# 行列转化
creat = np.array([[1,2]])
print(creat.T)
print(creat[:, np.newaxis])

print(np.concatenate((a,b,a,b), axis=0))

# 对矩阵的不等量分割(先尽可能均分，后面部分则逐行/列分配)
test = np.arange(1, 19).reshape((3,6))
print(np.array_split(test,4,axis=1))

np.random.uniform(-4, 4, 6)