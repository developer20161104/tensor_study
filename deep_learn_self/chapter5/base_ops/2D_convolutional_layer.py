import tensorflow as tf
import numpy as np


# x 为输入矩阵，k为卷积核
def corr2d(x, k):
    h, w = k.shape
    y = tf.Variable(tf.zeros(shape=[x.shape[0] - h + 1, x.shape[1] - w + 1]))

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            # 自定义的卷积操作由于对单个元素的赋值而无法进行梯度求取
            y[i, j].assign(tf.cast(tf.reduce_sum(x[i:i + h, j:j + w] * k), dtype=tf.float32))

    return y


class Con2D_self(tf.keras.layers.Layer):
    def __init__(self, unit):
        super().__init__()
        self.unit = unit

    # 卷积核对应着全连层中的w值
    def build(self, kernel_size):
        self.w = self.add_weight(
            name='w',
            shape=kernel_size,
            initializer=tf.random_normal_initializer
        )

        self.b = self.add_weight(
            name='b',
            shape=[1, ],
            initializer=tf.random_normal_initializer
        )

    def call(self, inputs):
        return corr2d(inputs, self.w) + self.b


def edge_detection():
    x = tf.Variable(tf.ones(shape=[6, 8]))
    x[:, 2:6].assign(tf.zeros_like(x[:, 2:6]))

    # print(x)
    k = tf.constant([[1, -1]], dtype=tf.float32)
    return corr2d(x, k)


if __name__ == '__main__':
    # edge_detection()

    # shapes,rows,cols,channels
    x = tf.Variable(tf.ones(shape=[1, 6, 8, 1]))
    y = tf.reshape(edge_detection(), shape=[1, 6, 7, 1])

    print(y)
    conv2d = tf.keras.layers.Conv2D(
        1,
        [1, 2]
    )

    pred_y = conv2d(x)
    print(pred_y)

    for i in range(10):
        with tf.GradientTape(watch_accessed_variables=False) as g:
            g.watch(conv2d.weights[0])

            pred_y = conv2d(x)
            l = (abs(pred_y - y)) ** 2
            dl = g.gradient(l, conv2d.weights[0])
            lr = 3e-2

            update = tf.multiply(lr, dl)
            update_weight = conv2d.get_weights()
            update_weight[0] = conv2d.weights[0] - update
            conv2d.set_weights(update_weight)

            if (i+1) %2 == 0:
                print('batch %d, loss %.3f' % (i + 1, tf.reduce_sum(l)))