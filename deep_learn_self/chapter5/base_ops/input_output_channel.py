import tensorflow as tf


# 实现内部的互相关(卷积)计算
def corr2d(x, k):
    h, w = k.shape

    if len(x.shape) <= 1:
        x = tf.reshape(x, [x.shape[0], 1])

    y = tf.Variable(tf.zeros([x.shape[0] - h + 1, x.shape[1] - w + 1]))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j].assign(tf.cast(tf.reduce_sum(x[i:i + h, j:j + w] * k), dtype=tf.float32))

    return y


# 含有多个输入通道的互相关计算（需要将结果在最后累加）
def corr2d_multi_in(x, k):
    return tf.reduce_sum([corr2d(x[i], k[i]) for i in range(x.shape[0])], axis=0)


def test_multi_cov2d():
    x = tf.constant([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    k = tf.constant([[[0, 1], [2, 3]],
                     [[1, 2], [3, 4]]])

    print(corr2d_multi_in(x, k))


def test():
    # 一个很有效的校验方法：计算经过每个层后当前特征矩阵的维度
    x = tf.random.normal(shape=[1, 28, 28, 1], mean=10, stddev=1)
    # print(x)
    conv2d = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=8,
        padding='same',
    )
    net = tf.keras.Sequential()
    net.add(conv2d)
    net.add(tf.keras.layers.MaxPool2D(2, 1))
    # 实际的filter维度为4*4*in_channels*32
    # 此处的in_channels的大小貌似会自己填充
    net.add(tf.keras.layers.Conv2D(32, 4, strides=2, padding='valid'))
    print(net(x).shape)


if __name__ == '__main__':
    # test_multi_cov2d()

    test()
