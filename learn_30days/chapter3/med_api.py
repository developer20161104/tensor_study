import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses, metrics, optimizers


# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return tf.strings.format("0{}", m)
        else:
            return tf.strings.format("{}", m)

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)


# sample number
n = 400

# create dataset
X = tf.random.uniform([n, 2], minval=-10, maxval=10)
w0 = tf.constant([[2.0], [-3.0]])
b0 = tf.constant([[3.0]])

Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0, stddev=0.01)

# create dataset pipe with inner function
# simpler than customer function
ds = tf.data.Dataset.from_tensor_slices((X, Y)) \
    .shuffle(buffer_size=100).batch(10) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# define methods directly
model = layers.Dense(units=1)
# use build to create the Variable
model.build(input_shape=(2,))
model.loss_func = losses.mean_squared_error
model.optimizer = optimizers.SGD(lr=1e-3)


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        pred_y = model(features)
        loss = model.loss_func(tf.reshape(labels, [-1]), tf.reshape(pred_y, [-1]))

    grads = tape.gradient(loss, model.variables)
    model.optimizer.apply_gradients(zip(grads, model.variables))

    return loss


# feature, label = next(ds.as_numpy_iterator())
# print(train_step(model, feature, label))

def train_model(epochs, model):
    for epoch in range(epochs):
        loss = tf.constant(0.0)
        for feature, label in ds:
            loss = train_step(model, feature, label)

        if not epoch % 50:
            printbar()
            tf.print("epoch =", epoch, "loss = ", loss)
            tf.print("w =", model.variables[0])
            tf.print("b =", model.variables[1])


# train_model(200, model)

# 正负样本数量
n_positive, n_negative = 2000, 2000

# positive samples
r_p = 5.0 + tf.random.truncated_normal([n_positive, 1], 0.0, 1.0)
theta_p = tf.random.uniform([n_positive, 1], 0.0, 2 * np.pi)
Xp = tf.concat([r_p * tf.cos(theta_p), r_p * tf.sin(theta_p)], axis=1)
Yp = tf.ones_like(r_p)

# negative samples
r_n = 12.0 + tf.random.truncated_normal([n_negative, 1], 0.0, 1.0)
theta_n = tf.random.uniform([n_negative, 1], 0.0, 2 * np.pi)
Xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
Yn = tf.zeros_like(r_n)

# concatenate double samples
X = tf.concat([Xp, Xn], axis=0)
Y = tf.concat([Yp, Yn], axis=0)

# create pipe
ds_dnn = tf.data.Dataset.from_tensor_slices((X, Y)) \
    .shuffle(buffer_size=4000).batch(100) \
    .prefetch(tf.data.experimental.AUTOTUNE)


class DNNModel(tf.Module):
    def __init__(self, name=None):
        super(DNNModel, self).__init__(name=name)
        self.dense1 = layers.Dense(4, activation="relu")
        self.dense2 = layers.Dense(8, activation="relu")
        self.dense3 = layers.Dense(1, activation="sigmoid")

    # 正向传播
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
    def __call__(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        y = self.dense3(x)
        return y


model = DNNModel()
model.loss_func = losses.binary_crossentropy
model.metric_func = metrics.binary_accuracy
model.optimizer = optimizers.Adam(lr=1e-3)


@tf.function
def train_step(model, feature, label):
    with tf.GradientTape() as tape:
        y_pred = model(feature)
        loss = model.loss_func(tf.reshape(label, [-1]), tf.reshape(y_pred, [-1]))

    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metric = model.metric_func(tf.reshape(label, [-1]), tf.reshape(y_pred, [-1]))

    return loss, metric


def train_model(model, epochs):
    for epoch in range(epochs):
        loss, metric = tf.constant(0.0), tf.constant(0.0)
        for feature, label in ds_dnn:
            loss, metric = train_step(model, feature, label)

        if not epoch % 10:
            printbar()
            tf.print("epoch =", epoch, "loss = ", loss, "accuracy = ", metric)


train_model(model, epochs=60)

# 结果可视化
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
ax1.scatter(Xp[:, 0].numpy(), Xp[:, 1].numpy(), c="r")
ax1.scatter(Xn[:, 0].numpy(), Xn[:, 1].numpy(), c="g")
ax1.legend(["positive", "negative"])
ax1.set_title("y_true")

Xp_pred = tf.boolean_mask(X, tf.squeeze(model(X) >= 0.5), axis=0)
Xn_pred = tf.boolean_mask(X, tf.squeeze(model(X) < 0.5), axis=0)

ax2.scatter(Xp_pred[:, 0].numpy(), Xp_pred[:, 1].numpy(), c="r")
ax2.scatter(Xn_pred[:, 0].numpy(), Xn_pred[:, 1].numpy(), c="g")
ax2.legend(["positive", "negative"])
ax2.set_title("y_pred")

plt.show()