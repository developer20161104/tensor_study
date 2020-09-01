import tensorflow as tf

# 比较方便，可以显示所有的求导过程，pytorch只能显示对叶子节点的求导
# 自动求取梯度
# x = tf.reshape(tf.constant(range(4), dtype=tf.float32), (4,1))
x = tf.constant(range(4), shape=[4, 1], dtype=tf.float32)

with tf.GradientTape() as t:
    t.watch(x)
    # 构建函数 2x**2
    y = 2 * tf.matmul(tf.transpose(x), x)

# 梯度获取
print(t.gradient(y, x))

with tf.GradientTape() as t:
    t.watch(x)

    y = x * x
    z = y * y

print(t.gradient(z, x))

# 查询帮助文档的操作
# 使用dir来查看可调用类或方法
print(dir(tf.random))

# 使用help函数来查看具体的使用方法
print(help(tf.matmul))
