import tensorflow as tf
import numpy as np

#
# 自定义模型与初始化器
#

net = tf.keras.models.Sequential()

net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
net.add(tf.keras.layers.Dense(10))


x = tf.random.uniform(shape=[2, 20])
# print(x)
y = net(x)
# print(y)

# print(net.weights)


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1),
            bias_initializer=tf.zeros_initializer()
        )

        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, input):
        output = self.d1(input)
        output = self.d2(output)

        return output


def call_customer():
    net = Linear()
    print(net(x))
    print(net.weights)


# 自定义化的参数初始化器
def my_init():
    return  tf.keras.initializers.zeros()


if __name__ == '__main__':
    call_customer()