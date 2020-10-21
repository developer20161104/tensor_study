#
# 自动微分机制：即利用梯度磁带来求取导数
#

import numpy as np
import tensorflow as tf

x = tf.Variable(0.0, name='x', dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    # 构建函数式
    y = a * tf.pow(x, 2) + b * x + c

dy_dx = tape.gradient(y, x)

print(dy_dx)

# 对常量张量也可以求导，需要增加watch
with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a * tf.pow(x, 2) + b * x + c

dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_da)
print(dy_dc)


# 将eager模式转化为Autograph，提升效率
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    x = tf.cast(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)

    # 此处相当于静态图模式，因此无法像eager模式一样提取numpy
    return dy_dx, y


# 可以在脱离静态图模式后进行提取操作
print(f(0.0)[1].numpy())

# 最小值求取
# 方法一：使用apply_gradients
optimizer = tf.keras.optimizers.SGD(lr=0.01)

for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c

    grads = tape.gradient(y, x)
    optimizer.apply_gradients([(grads, x)])

tf.print('y=', y, ' x=', x)


# 方法二：使用minimze
# 此处的f是无参数的
@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return (y)


for _ in range(1000):
    optimizer.minimize(f, [x])

tf.print('y=', y, ' x=', x)
