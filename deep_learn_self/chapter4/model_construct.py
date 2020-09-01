import tensorflow as tf
import numpy as np


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def __call__(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


x = tf.random.uniform(shape=(2, 20))
model = MLP()
# print(model(x), MLP()(x))


class FancyMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(tf.random.uniform(shape=[20, 20]))

        self.dense = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)

    def __call__(self, inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(tf.matmul(x, self.rand_weight)+1)

        x = self.dense(x)

        # 欧几里得范数：即平方和开根号
        while tf.norm(x) > 1:
            x /= 2
        if tf.norm(x) < 0.8:
            x *= 10

        return tf.reduce_sum(x)


model1 = FancyMLP()
# print(model1(x).numpy())

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(2,20,2)))
net.add(tf.keras.layers.Dense(256,activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(stddev=100)))
net.add(tf.keras.layers.Dense(10))

print(net.weights)