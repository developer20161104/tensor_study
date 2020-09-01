import tensorflow as tf
from matplotlib import pyplot as plt
import random
import numpy as np

# # 神奇的广播机制，将标量直接转化为矢量并加入计算，极大提升速率
# a = tf.ones((3,))
# print(a+10)

# create dataset
num_input = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.3

features = tf.random.normal((num_examples, num_input), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 施加一定噪声
labels += tf.random.normal((num_examples,), stddev=0.01)


# print(features, "label: ", labels)

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


# 显示图像
# set_figsize()
# plt.scatter(features[:, 1], labels, 1)

# 数据的读取
# 通过打乱下标实现shuffle读取，这一点有点意思
def data_iter(batch_size, features, labels):
    total = len(labels)
    index_total = list(range(total))
    # 打乱下标
    random.shuffle(index_total)

    for i in range(0, total, batch_size):
        j = index_total[i:min(batch_size + i, total)]
        # 使用生成器减少内存开销,gather抽取不连续的区间
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)


# batch_size = 10
# for x,y in data_iter(batch_size, features, labels):
#     print(x,y)

# 初始化模型参数，将w设置为均值为0，标准差为1的正态随机数，b设置为0
# 注意到需要进行梯度计算，因此需要转为Variable类型
w = tf.Variable(tf.random.normal((num_input, 1), stddev=0.01))
b = tf.Variable(tf.zeros((1,)))


# 模型定义部分
def linear_regression(X, w, b):
    return tf.matmul(X, w) + b


# 定义损失函数
def loss_func(pred_y, y):
    return (pred_y - tf.reshape(y, pred_y.shape)) ** 2 / 2


# 定义优化器 mini-batch SGD
def mini_sgd(params, lr, batch_size, grads):
    for i, params in enumerate(params):
        # 计算各参数经过梯度优化的值
        params.assign_sub(lr * grads[i] / batch_size)


if __name__ == '__main__':
    # 进行模型的训练
    lr = 0.01
    num_epochs = 4
    batch_size = 10
    loss_epoch = []

    for epoch in range(num_epochs):
        # mini_batch training
        for x, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as t:
                t.watch([w, b])

                # 需要转为标量进行求导
                l = tf.reduce_sum(loss_func(linear_regression(x, w, b), y))
            # 获取梯度
            grads = t.gradient(l, [w, b])
            mini_sgd([w, b], lr, batch_size, grads)

        # 测试
        train_epoch = loss_func(linear_regression(features, w, b), labels)
        loss_epoch.append(tf.reduce_mean(train_epoch))

    # 绘制loss折线图
    plt.plot(tf.range(num_epochs), np.array(loss_epoch))

    test_x = tf.transpose(tf.sort(tf.random.uniform([1, 30], 0.0, 2.0)))

    # plt.plot(test_x, linear_regression(tf.concat([test_x, test_x], axis=1), tf.constant(true_w, shape=[2, 1]),
    # tf.constant(true_b)), 'r--') plt.plot(test_x, linear_regression(tf.concat([test_x, test_x], axis=1), w, b), 'g--')

    plt.show()
