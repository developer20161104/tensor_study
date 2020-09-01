# 权重衰减等价于L2正则化，为模型的损失函数添加惩罚项使得学出的模型参数值较小
# 采用的是高斯线性回归实验

import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers, regularizers
import numpy as np
import matplotlib.pyplot as plt
import linear_reg.linear_regression as aux
from solutions_overfitting.under_overfitting import semilogy

n_train, n_test, num_input = 20, 100, 200
true_w, true_b = tf.ones(shape=[num_input, 1]) * 0.01, 0.05

features = tf.random.normal(shape=[n_train + n_test, num_input], stddev=0.01)
labels = tf.matmul(features, true_w) + true_b
labels += tf.random.normal(mean=0.1, shape=labels.shape)

# 数据集的切割
train_feature, train_label, test_feature, test_label = features[:n_train, :], labels[:n_train], features[n_train:,
                                                                                                :], labels[n_train:]


def init_params():
    w = tf.Variable(tf.random.normal(shape=[num_input,1], stddev=0.1))
    b = tf.Variable(tf.zeros(shape=[1, ]))
    return [w, b]


# 定义惩罚项
def l2_penalty(w):
    return tf.reduce_sum((w ** 2)) / 2


batch_size = 1
num_epoch = 100
lr = 0.003
net, loss = aux.linear_regression, aux.loss_func
optimizer = optimizers.SGD(lr)
train_iter = tf.data.Dataset.from_tensor_slices((train_feature, train_label)).batch(batch_size).shuffle(batch_size)


def fit_and_plot(lambd):
    w, b = init_params()

    train_ls, test_ls = [], []
    for _ in range(num_epoch):
        for x, y in train_iter:
            with tf.GradientTape() as t:
                # 添加惩罚项
                l = loss(net(x, w, b), y) + lambd*l2_penalty(w)
            grads = t.gradient(l, [w, b])
            aux.mini_sgd([w,b], lr, batch_size, grads)

        train_ls.append(np.mean(loss(net(train_feature,w,b), train_label)))
        test_ls.append(np.mean(loss(net(test_feature, w,b), test_label)))

    semilogy(range(1, num_epoch + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epoch + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', tf.norm(w).numpy())


def fit_and_plot_keras(wd, lr=1e-3):
    model = tf.keras.Sequential([
        layers.Dense(1,
                     kernel_regularizer=regularizers.l2(wd),
                     bias_regularizer=None)
    ])

    model.compile(
        optimizer=optimizers.SGD(lr),
        loss=tf.keras.losses.MeanSquaredError()
    )

    history = model.fit(
        train_feature,
        train_label,
        epochs=100,
        batch_size=1,
        validation_data=(test_feature, test_label),
        validation_freq=1,
        verbose=0
    )

    train_ls = history.history['loss']
    test_ls = history.history['val_loss']
    semilogy(range(1, num_epoch + 1), train_ls, 'epochs', 'loss',
                     range(1, num_epoch + 1), test_ls, ['train', 'test'])

    print('L2 norm of w:', tf.norm(model.get_weights()[0]).numpy())


if __name__ == '__main__':
    # fit_and_plot(0)
    # fit_and_plot(3)

    fit_and_plot_keras(0)
    fit_and_plot_keras(4)