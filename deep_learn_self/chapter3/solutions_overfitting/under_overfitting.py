import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用的样本函数 y = 1.2x - 3.4x**2 + 5.6x**3 + 5 + eps
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = tf.random.normal(shape=[n_train + n_test, 1])
poly_features = tf.concat([features, tf.pow(features, 2), tf.pow(features, 3)], 1)
print(poly_features.shape)

# 矢量计算确实挺方便
labels = tf.matmul(poly_features, tf.reshape(true_w, [len(true_w), 1])) + true_b
# labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]+ true_w[2] * poly_features[:, 2] + true_b)
labels += tf.random.normal(stddev=0.1, shape=labels.shape)
print(labels.shape)


# 图像绘制部分
def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)

    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# model training
num_epochs, loss_func = 100, tf.losses.MeanSquaredError()


def fit_and_plot(train_features, test_features, train_label, test_labels):
    net = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    batch_size = min(10, train_label.shape[0])
    train_iter = tf.data.Dataset.from_tensor_slices((train_features, train_label)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(batch_size)

    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for x, y in train_iter:
            with tf.GradientTape() as t:
                l = loss_func(y, net(x))

            grads = t.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

        train_ls.append(np.mean(loss_func(train_label, net(train_features))))
        test_ls.append(np.mean(loss_func(test_labels, net(test_features))))
        # train_ls.append(loss_func(train_label, net(train_features)).numpy().mean())
        # test_ls.append(loss_func(test_labels, net(test_features)).numpy().mean())

    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.get_weights()[0],
          '\nbias:', net.get_weights()[1])


if __name__ == '__main__':
    # 注意选取的方式与一般的不一致，由于采用的是[,]形式，因此，第一维操作表示为对行的操作，第二维为对列的操作
    fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
                 labels[:n_train], labels[n_train:])


    # underfitting
    # 直接使用一阶的进行拟合就会出现这个情况
    fit_and_plot(features[:n_train, :], features[n_train:, :],
                 labels[:n_train], labels[n_train:])

    # overfitting
    # 训练样本过少
    fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
                 labels[n_train:])