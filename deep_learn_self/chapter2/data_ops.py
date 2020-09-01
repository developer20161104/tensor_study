import tensorflow as tf
import numpy as np

print(tf.__version__)

# 由于eager模式的存在，因此可以直接返回张量，看到计算的结果
# tensor 与多维数组很像，并提供了GPU计算与自动求梯度的功能
x = tf.constant(range(12))
print(x.shape)

# 只需要拟定一个维度来进行重构
x = tf.reshape(x, (3, -1))
print(x)

# 以正态分布随机生成样本
print(tf.random.normal(shape=[3, 4], mean=0, stddev=1))

y = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 进行矩阵的乘法，在这点上与numpy一样，直接使用*为点乘
y = tf.cast(y, tf.int32)
print(tf.matmul(x, tf.transpose(y)))

# 连结矩阵
print(tf.concat([x, y], axis=0))
print(tf.concat([x, y], axis=1))

# 内部元素逐一判别
print(tf.equal(x, y))

# operations on different dim of tensors
# 会触发广播机制
# 将操作后的张量拓展为max(col)与max(row)的维度再进行计算
a = tf.reshape(tf.constant(range(3)), (1, 3))
b = tf.reshape(tf.constant(range(2)), (2, 1))

print(a + b)

# 索引操作
x = tf.reshape(tf.constant(range(1, 13)), (3, 4))

# 必须要转化为Variable才能进行索引操作,由于矩阵特点，可以直接对多行/列进行操作
# 只有Variable类型可以保存梯度的信息，因此在自动求导时需要考虑类型的转化
x = tf.Variable(x)
# print(x[1, 2].assign(9))
print(x[:2, :].assign(tf.ones(x[:2, :].shape, dtype=tf.int32) * 12))

# 与numpy的相互转化
# to tensor
p = np.arange(1, 13).reshape((3,4))
temp = tf.constant(p)
print(temp)

# to array
print(np.array(temp))