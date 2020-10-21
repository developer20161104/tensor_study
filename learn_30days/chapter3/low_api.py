import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 时间线打印
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    # 时间规范化
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minite)
                                     , timeformat(second)], separator=':')

    tf.print("==========" * 8 + timestring)


#######################################################################
####################### linear regression model #######################
#######################################################################
# 样本数量
# n = 400
#
# # 获取数据集
# X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])
#
# # @ 为矩阵的乘法
# Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2.0)


# plt.figure(figsize = (12,5))
# ax1 = plt.subplot(121)
# ax1.scatter(X[:,0],Y[:,0], c = "b")
# plt.xlabel("x1")
# plt.ylabel("y",rotation = 0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(X[:,1],Y[:,0], c = "g")
# plt.xlabel("x2")
# plt.ylabel("y",rotation = 0)
# plt.show()


# 构建数据管道迭代器
def data_iter(features, labels, batch_size=8):
    num_examples = len(features)

    # 利用下标来打乱序列
    indices = list(range(num_examples))
    np.random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        # 截断操作
        index = indices[i:min(batch_size + i, num_examples)]

        # 利用gather来获取对应位置的数值
        yield tf.gather(features, index), tf.gather(labels, index)


# for x, y in data_iter(X, Y):
#     print(x, y)

# 模型定义
w = tf.Variable(tf.random.normal(w0.shape))
b = tf.Variable(tf.zeros_like(b0, dtype=tf.float32))


# 定义模型
class LinearRegression:
    def __call__(self, x):
        return x @ w + b

    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean((y_pred - y_true) ** 2 / 2)


model = LinearRegression()


# 模型训练
# 转化为Autograph贼快
@tf.function
def train_step(model, features, labels, lr=0.01):
    with tf.GradientTape() as tape:
        pred = model(features)
        loss = model.loss_func(labels, pred)

    dw, db = tape.gradient(loss, [w, b])
    w.assign(w - dw * lr)
    b.assign(b - db * lr)

    return loss


# batch_size = 10
# (features, labels) = next(data_iter(X, Y, batch_size))
# print(train_step(model, features, labels))
def train_model(model, epochs, batch_size=10):
    for epoch in range(1, epochs + 1):
        for features, labels in data_iter(X, Y, batch_size):
            loss = train_step(model, features, labels)

        if not epoch % 50:
            printbar()
            tf.print("epoch =", epoch, "loss = ", loss)
            tf.print("w =", w)
            tf.print("b =", b)


# 模型的训练与可视化
# train_model(model, 200)
#
# plt.figure(figsize=(12, 5))
# ax1 = plt.subplot(121)
# ax1.scatter(X[:, 0], Y[:, 0], c="b", label="samples")
# ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], "-r", linewidth=5.0, label="model")
# ax1.legend()
# plt.xlabel("x1")
# plt.ylabel("y", rotation=0)
#
# ax2 = plt.subplot(122)
# ax2.scatter(X[:, 1], Y[:, 0], c="g", label="samples")
# ax2.plot(X[:, 1], w[1] * X[:, 1] + b[0], "-r", linewidth=5.0, label="model")
# ax2.legend()
# plt.xlabel("x2")
# plt.ylabel("y", rotation=0)
#
# plt.show()


#######################################################################
####################### DNN classification model ######################
#######################################################################
# 样本的生成
n_pos, n_neg = 2000, 2000

r_p = 5.0 + tf.random.truncated_normal([n_pos, 1], 0.0, 1.0)
theta_p = tf.random.uniform([n_pos, 1], 0.0, 2 * np.pi)
Xp = tf.concat([r_p * tf.cos(theta_p), r_p * tf.sin(theta_p)], axis=1)
Yp = tf.ones_like(r_p)

r_n = 10.0 + tf.random.truncated_normal([n_neg, 1], 0.0, 1.0)
theta_n = tf.random.uniform([n_neg, 1], 0.0, 2 * np.pi)
Xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
Yn = tf.zeros_like(r_n)

X = tf.concat([Xp, Xn], axis=0)
Y = tf.concat([Yp, Yn], axis=0)


# # 管道构建
# def data_iter_DNN(features, labels, batch_size=10):
#     num_exp = len(features)
#     indices = list(range(num_exp))
#
#     np.random.shuffle(indices)
#
#     for i in range(0, num_exp, batch_size):
#         index = indices[i:min(batch_size+i, num_exp)]
#         yield tf.gather(features, index), tf.gather(labels, index)



class DNN(tf.Module):
    def __init__(self, name=None):
        super(DNN, self).__init__(name=name)
        self.w1 = tf.Variable(tf.random.truncated_normal([2, 4]), dtype=tf.float32)
        self.b1 = tf.Variable(tf.zeros([1, 4]), dtype=tf.float32)
        self.w2 = tf.Variable(tf.random.truncated_normal([4, 8]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([1, 8]), dtype=tf.float32)
        self.w3 = tf.Variable(tf.random.truncated_normal([8, 1]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([1, 1]), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.nn.relu(x @ self.w1 + self.b1)
        x = tf.nn.relu(x @ self.w2 + self.b2)
        y = tf.nn.sigmoid(x @ self.w3 + self.b3)
        return y

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    ])
    def loss_func(self, y_true, y_pred):
        eps = 1e-7

        # 限制预测值的范围，防止出现 log 0
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # 交叉熵
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(bce)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    ])
    def metric_func(self, y_ture, y_pred):
        # condition,a,b
        # 返回的是 b, b中的元素满足condition的位置由a中对应位置的元素来替代
        # 即满足 y_pred>0.5 条件的位置置1，其余为0
        y_pred = tf.where(
            y_pred > 0.5, tf.ones_like(y_pred, dtype=tf.float32),
            tf.zeros_like(y_pred, dtype=tf.float32)
        )

        return tf.reduce_mean(1 - tf.abs(y_ture - y_pred))

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 1], dtype=tf.float32)])
    def metric_func2(self, y_true, y_pred):
        y_pred = tf.where(y_pred > 0.5, tf.ones_like(y_pred, dtype=tf.float32),
                          tf.zeros_like(y_pred, dtype=tf.float32))
        acc = tf.reduce_mean(1 - tf.abs(y_true - y_pred))
        return acc

# 测试模型结构
# model = DNN()
# batch_size = 10
# (features,labels) = next(data_iter(X_1,Y_1,batch_size))
#
# predictions = model(features)
#
# loss = model.loss_func(labels,predictions)
# metric = model.metric_func2(labels,predictions)
#
# tf.print("init loss:",loss)
# tf.print("init metric",metric)


# 模型训练
@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        pred = model(features)

        loss = model.loss_func(labels, pred)

    grads = tape.gradient(loss, model.trainable_variables)

    # 模型参数较多时的梯度下降
    for p, dp in zip(model.trainable_variables, grads):
        p.assign(p - 0.001 * dp)

    metric = model.metric_func(labels, pred)

    return loss, metric


def train_model(model, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in data_iter(X, Y, 100):
            loss, metric = train_step(model, features, labels)
        if epoch % 100 == 0:
            printbar()
            tf.print("epoch =", epoch, "loss = ", loss, "accuracy = ", metric)


train_model(DNN(), epochs=600)
# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(Xp[:, 0], Xp[:, 1], c="r")
ax1.scatter(Xn[:, 0], Xn[:, 1], c="g")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

Xp_pred = tf.boolean_mask(X, tf.squeeze(model(X) > 0.5), axis=0)
Xn_pred = tf.boolean_mask(X, tf.squeeze(model(X) <= 0.5), axis=0)

ax2.scatter(Xp_pred[:, 0], Xp_pred[:, 1], c="r")
ax2.scatter(Xn_pred[:, 0], Xn_pred[:, 1], c="g")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")

plt.show()
