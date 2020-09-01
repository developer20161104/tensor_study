#
# 计算卷积后的维度的公式 (n-k+p+s) / s
# n为输入维，k为卷积核维，p为填充维（一般只有k-1或者0），s为步长维
#
import tensorflow as tf


def comp_conv2d(conv2d, x):
    # 还能这么组合吗？
    x = tf.reshape(x, shape=[1, ] + x.shape + [1, ])

    y = conv2d(x)
    return tf.reshape(y, y.shape[1:3])


def padding_test():
    return tf.keras.layers.Conv2D(
        # 卷积核的数量
        filters=1,
        # 卷积核的大小
        kernel_size=3,
        # 填充的方式
        padding='same'
    )


def striding_test():
    return tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=[3, 5],
        padding='valid',
        strides=[3, 4]
    )


if __name__ == '__main__':
    # conv2d = padding_test()
    conv2d = striding_test()

    x = tf.random.uniform(shape=[8, 8])
    print(comp_conv2d(conv2d, x).shape)
