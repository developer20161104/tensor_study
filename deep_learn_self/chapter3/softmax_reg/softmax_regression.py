# softmax 适合于离散数值的预测与训练
# 输出单元为预测的种类数目，转化为分类问题


import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
# 导入数据集
from tensorflow.keras.datasets import fashion_mnist

# 预设置的数据集，提取数据比较方便
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# print(len(x_train), len(x_test))
# print(x_train[1].shape, x_test[0].dtype)


# 设置类别标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 图像绘制函数
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(24, 24))

    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img)
        f.set_title(lbl)

        f.axes.get_yaxis().set_visible(False)
        f.axes.get_xaxis().set_visible(False)

    plt.show()


# x,y = [], []
# for i in range(100, 110):
#     x.append(x_train[i])
#     y.append(y_train[i])
# show_fashion_mnist(x,get_fashion_mnist_labels(y))

# 将类型进行转化，并设置小批量读取，过程与之前的一致
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

# 设置除法的目的个人感觉是为了进行规约
x_train = tf.cast(x_train, tf.float32) / 255
x_test = tf.cast(x_test, tf.float32) / 255

train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# 模型参数的初始化
num_input = 784  # 28*28
num_output = 10  # 10 class

w = tf.Variable(tf.random.normal((num_input, num_output), stddev=0.1, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_output, dtype=tf.float32))


# 实现softmax,这个-1有点牛皮哦，可以处理任意维度
def softmax_cus(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


# 模型定义
def net(x):
    # 即一层全连计算加一层softmax
    logits = tf.matmul(tf.reshape(x, shape=(-1, w.shape[0])), w) + b
    return softmax_cus(logits)


# 定义损失函数
def loss_func(pred_y, y):
    # 初始的标签值为一个整数，需要进行转化，再放入one-hot编码中
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=pred_y.shape[-1])
    # 对one-hot编码进行转置以及数据类型变化
    y = tf.cast(tf.reshape(y, shape=[-1, pred_y.shape[-1]]), dtype=tf.int32)

    # 获取该标签的预测概率，一共有10种类别，找到为1的类别的概率返回
    return -tf.math.log(tf.boolean_mask(pred_y, y) + 1e-8)


# 准确率的计算：判断预测标签是否与真实标签一致即可
def accuracy(pred_y, y):
    # 对多个样本求取均值，注意类型必须为int
    # 先寻找列中的最大值所在下标，再与真实下标进行比较
    return np.mean((tf.cast(tf.argmax(pred_y, axis=1), dtype=tf.int32) == y))


# 测试初始的网络计算的精确度
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for (_, (x, y)) in enumerate(data_iter):
        y = tf.cast(y, dtype=tf.int32)
        # 注意argmax比较的为行上的数值
        acc_sum += np.sum(tf.cast(tf.argmax(net(x), axis=1), dtype=tf.int32) == y)
        n += y.shape[0]

    return acc_sum / n


# 初始网络的准确率接近0.1
# print(evaluate_accuracy(test_iter, net))


# 模型的训练与测试
lr = 0.1
epochs = 10


def train_softmax_regression(net, train_iter, test_iter, loss, epochs, batch_size, params=None, lr=0.1, trainer=None):
    res = [[] for _ in range(3)]

    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for (x, y) in train_iter:
            with tf.GradientTape() as t:
                # 前向计算
                pred_y = net(x)
                l = tf.reduce_sum(loss(pred_y, y))
            # 获取反向的梯度
            grads = t.gradient(l, params)

            if trainer is None:
                # 使用之前的实现的小批量sgd
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                # 直接使用优化器中的mini-batch SGD
                # 一个梯度对应着一个参数
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))

            y = tf.cast(y, dtype=tf.int32)
            # 训练集损失量与准确数量
            train_l_sum += l.numpy()
            train_accuracy = accuracy(pred_y, y)
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
            epoch + 1, train_l_sum / n, train_accuracy, test_acc))
        res[0].append(train_l_sum / n)
        res[1].append(train_accuracy)
        res[2].append(test_acc)

    return res


# loss_s, train_s, test_s = train_softmax_regression(net, train_iter, test_iter, loss_func, epochs, batch_size, [w, b],
#                                                    lr)
# x_axis = tf.range(1.0, len(loss_s) + 1)
# plt.plot(x_axis, np.array(loss_s), '--r', label='loss')
# plt.plot(x_axis, np.array(train_s), '--g', label='train')
# plt.plot(x_axis, np.array(test_s), '--b', label='test')
# plt.legend(loc='upper left')
#
# predict_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
# # 显示预测的结果
# x, y = iter(predict_iter).__next__()
#
# true_labels = get_fashion_mnist_labels(y.numpy())
# pred_labels = get_fashion_mnist_labels(tf.argmax(net(x), axis=1))
#
# titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
# show_fashion_mnist(x[0:9], titles[0:9])
#
# plt.show()
