import tensorflow as tf
import numpy as np


# 自定义层级
class CenteredLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


def get_net():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20),
        CenteredLayer()
    ])


class myDense(tf.keras.layers.Layer):
    def __init__(self, unit):
        super().__init__()
        self.unit = unit

    # input shape 为初始放入的形状
    # 获取输入维度，以便动态构建需要的算子
    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=[input_shape[-1], self.unit], initializer=tf.random_normal_initializer)

        self.b = self.add_weight(name='b',
                                 shape=self.unit, initializer=tf.zeros_initializer)
        # print(self.w.shape, self.b.shape)
        # print(tf.matmul(self.w, tf.reshape(self.b, shape=[self.b.shape[0], 1])).shape)

    def call(self, input):
        pred_y = tf.matmul(input, self.w) + self.b
        return pred_y


if __name__ == '__main__':
    x = tf.random.uniform(shape=[2, 10])

    # net = get_net()
    # y = net(x)
    # print(y)
    # print(tf.reduce_mean(y))

    dense = myDense(3)
    dense(x)
    print(dense)
    print(dense.get_weights())