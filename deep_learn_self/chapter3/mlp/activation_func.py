# %matplotlib inline
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import random
import matplotlib


#
# 一些常用的激活函数演示
# 关于激活函数：作为层级连接中的非线性映射部分，使得层级的叠加不再只是单纯的仿射变换
#

# pycharm 不支持Magic函数
# def use_svg_display():
#     %config InlineBackend.figure_format = 'svg'

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


def xyplot(x_vals, y_vals, name):
    # 取消同一个图显示所有曲线
    plt.figure()
    # plt.cla()

    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')


# ****************** ReLU function ******************
#                   ReLU(x) = max(x,0)
x = tf.Variable(tf.range(-8, 8, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
xyplot(x, y, 'ReLU')

# gradient of ReLU
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.nn.relu(x)

# 加入的为因变量与自变量（需要求导的参数，一般为w与b）
dy_dx = t.gradient(y, x)
xyplot(x, dy_dx, 'grad of ReLU')

# ****************** sigmoid function ******************
#                 sigmoid(x) = 1 / (1 + exp(-x))
y = tf.nn.sigmoid(x)
xyplot(x, y, 'sigmoid')

# gradient of sigmoid
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.sigmoid(x)
xyplot(x, t.gradient(y, x), 'grad of sigmoid')

# ****************** tanh function ******************
#                   tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
y = tf.nn.tanh(x)
xyplot(x, y, 'tanh')

# gradient of tanh
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.tanh(x)
xyplot(x, t.gradient(y, x), 'grad of tanh')

plt.show()
