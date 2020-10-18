#
# TensorFlow程序 = 张量数据结构 + 计算图算法语言
# 张量从行为特性来看，有两种类型的张量，常量constant和变量Variable
# 常量的值在计算图中不可以被重新赋值，变量可以在计算图中用assign等算子重新赋值
#

import numpy as np
import tensorflow as tf

# 与numpy的关系对应
i = tf.constant(1)  # tf.int32 类型常量
l = tf.constant(1, dtype=tf.int64)  # tf.int64 类型常量
f = tf.constant(1.23)  # tf.float32 类型常量
d = tf.constant(3.14, dtype=tf.double)  # tf.double 类型常量
s = tf.constant("hello world")  # tf.string类型常量
b = tf.constant(True)  # tf.bool类型常量

print(tf.int64 == np.int64)
print(tf.float64 == np.float64)
print(tf.double == np.float64)

vector = tf.constant([1.0,2.0,3.0,4.0]) #向量，1维张量

# 输出维度数值
print(tf.rank(vector))
# print(vector.shape) 输出具体维度
print(np.ndim(vector))

# unicode 编码处理
u = tf.constant(u"你好 世界")
print(u.numpy())
print(u.numpy().decode('utf-8'))

# 关于变量张量
# 变量的值可以改变，可以通过assign, assign_add等方法给变量重新赋值
v = tf.Variable([1.0,2.0],name = "v")

v.assign_add([1.0, 1.0])
print(v)