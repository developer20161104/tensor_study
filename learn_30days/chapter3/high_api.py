import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, optimizers


# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)


def train_linear_regression():
    # sample number
    n = 400

    # create dataset
    X = tf.random.uniform([n, 2], minval=-10, maxval=10)
    w0 = tf.constant([[2.0], [-3.0]])
    b0 = tf.constant([[3.0]])

    Y = X @ w0 + b0 + tf.random.normal([n, 1], mean=0.0, stddev=2)

    # use high level api
    model = models.Sequential()
    model.add(layers.Dense(1, input_shape=(2,)))
    # model.summary()


    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    model.fit(
        X, Y,
        batch_size=10,
        epochs=200
    )

    w, b = model.variables

    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)
    ax1.scatter(X[:, 0], Y[:, 0], c="b", label="samples")
    ax1.plot(X[:, 0], w[0] * X[:, 0] + b[0], "-r", linewidth=5.0, label="model")
    ax1.legend()
    plt.xlabel("x1")
    plt.ylabel("y", rotation=0)

    ax2 = plt.subplot(122)
    ax2.scatter(X[:, 1], Y[:, 0], c="g", label="samples")
    ax2.plot(X[:, 1], w[1] * X[:, 1] + b[0], "-r", linewidth=5.0, label="model")
    ax2.legend()
    plt.xlabel("x2")
    plt.ylabel("y", rotation=0)

    plt.show()


def train_DNN():
    # create positive & negative sample
    n_positive, n_negative = 2000, 2000
    n = n_negative + n_positive

    # 生成正样本, 小圆环分布
    r_p = 5.0 + tf.random.truncated_normal([n_positive, 1], 0.0, 1.0)
    theta_p = tf.random.uniform([n_positive, 1], 0.0, 2 * np.pi)
    Xp = tf.concat([r_p * tf.cos(theta_p), r_p * tf.sin(theta_p)], axis=1)
    Yp = tf.ones_like(r_p)

    # 生成负样本, 大圆环分布
    r_n = 8.0 + tf.random.truncated_normal([n_negative, 1], 0.0, 1.0)
    theta_n = tf.random.uniform([n_negative, 1], 0.0, 2 * np.pi)
    Xn = tf.concat([r_n * tf.cos(theta_n), r_n * tf.sin(theta_n)], axis=1)
    Yn = tf.zeros_like(r_n)

    # based on sample
    X = tf.concat([Xp, Xn], axis=0)
    Y = tf.concat([Yp, Yn], axis=0)

    # based on feature and then shuffle
    data = tf.concat([X, Y], axis=1)
    data = tf.random.shuffle(data)

    X = data[:, :2]
    Y = data[:, 2:]

    ds_train = tf.data.Dataset.from_tensor_slices((X[0:n*3//4, :], Y[0:n*3//4, :])) \
        .shuffle(buffer_size= 1000).batch(20) \
        .prefetch(tf.data.experimental.AUTOTUNE).cache()

    ds_valid = tf.data.Dataset.from_tensor_slices((X[n*3//4:, :], Y[n*3//4:, :])) \
        .batch(20).prefetch(tf.data.experimental.AUTOTUNE).cache()

    model = models.Sequential(
        [
            layers.Dense(4, 'relu', name='dense1'),
            layers.Dense(8, 'relu', name='dense2'),
            layers.Dense(1, 'sigmoid', name='dense3'),
        ]
    )

    # custom training mode
    optimizer = optimizers.Adam(learning_rate=0.01)
    loss_func = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_metric = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(model, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features)
            loss = loss_func(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss.update_state(loss)
        train_metric.update_state(labels, predictions)

    @tf.function
    def valid_step(model, features, labels):
        predictions = model(features)
        batch_loss = loss_func(labels, predictions)
        valid_loss.update_state(batch_loss)
        valid_metric.update_state(labels, predictions)

    def train_model(model, ds_train, ds_valid, epochs):
        for epoch in tf.range(1, epochs + 1):
            for features, labels in ds_train:
                train_step(model, features, labels)

            for features, labels in ds_valid:
                valid_step(model, features, labels)

            logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

            if epoch % 100 == 0:
                printbar()
                tf.print(tf.strings.format(logs,
                                           (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                            valid_metric.result())))

            train_loss.reset_states()
            valid_loss.reset_states()
            train_metric.reset_states()
            valid_metric.reset_states()

    train_model(model, ds_train, ds_valid, 1000)

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


if __name__ == '__main__':
    train_DNN()