import tensorflow as tf
import numpy as np


def drop_out(x, drop_prob):
    assert 0 <= drop_prob <= 1

    keep = 1 - drop_prob
    if keep == 0:
        return tf.zeros_like(x)

    # 以均匀分布来产生随机数进行筛选
    mask = tf.random.uniform(shape=x.shape, minval=0, maxval=1) < keep
    return tf.cast(mask, dtype=tf.float32) * tf.cast(x, dtype=tf.float32)


def net(x, is_training=False):
    x = tf.reshape(x, shape=[-1, num_inputs])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    if is_training:
        h1 = drop_out(h1, drop_prob1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    if is_training:
        h2 = drop_out(h2, drop_prob2)

    return tf.nn.softmax(tf.matmul(h2, W3) + b3)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y in data_iter:
        y = tf.cast(y, tf.int32)
        acc_sum += np.sum(tf.cast(tf.argmax(net(x), axis=1), dtype=tf.int32) == y)
        n += y.shape[0]
    return acc_sum / n


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    global sample_grads
    for epoch in range(num_epochs):
        train_l, train_acc_count, n = 0.0, 0, 0
        for x, y in train_iter:
            with tf.GradientTape() as t:
                # 得到的是每个类的预测概率，需要进行转化
                pred_y = net(x, is_training=True)
                l = tf.reduce_sum(loss(pred_y, tf.one_hot(y, depth=10, axis=-1, dtype=tf.float32)))

            grads = t.gradient(l, params)
            if trainer is None:
                sample_grads = grads
                for index, param in enumerate(params):
                    param.assign_sub(grads[index]*lr)
            else:
                trainer.apply_gradients(zip(grads, params))

            y = tf.cast(y, tf.int32)
            train_l += l.numpy()
            train_acc_count += np.sum(tf.cast(tf.argmax(pred_y, axis=1), dtype=tf.int32) == y)
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l / n, train_acc_count / n, test_acc))


def main_keras(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        batch_size= 256,
        epochs=5,
        validation_data=(x_test, y_test),
        validation_freq=1,
    )


if __name__ == '__main__':
    # x = tf.reshape(tf.range(16), shape=[2, 8])
    # print(drop_out(x, 0.5))

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    W1 = tf.Variable(tf.random.normal(stddev=0.01, shape=(num_inputs, num_hiddens1)))
    b1 = tf.Variable(tf.zeros(num_hiddens1))
    W2 = tf.Variable(tf.random.normal(stddev=0.1, shape=(num_hiddens1, num_hiddens2)))
    b2 = tf.Variable(tf.zeros(num_hiddens2))
    W3 = tf.Variable(tf.random.truncated_normal(stddev=0.01, shape=(num_hiddens2, num_outputs)))
    b3 = tf.Variable(tf.zeros(num_outputs))

    params = [W1, b1, W2, b2, W3, b3]

    loss = tf.losses.CategoricalCrossentropy()
    num_epochs, lr, batch_size = 5, 0.5, 256
    drop_prob1, drop_prob2 = 0.2, 0.5

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = tf.cast(x_train, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
    x_test = tf.cast(x_test, tf.float32) / 255  # 在进行矩阵相乘时需要float型，故强制类型转换为float型
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)

    # by keras
    # main_keras(x_train,y_train,x_test,y_test)