import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from softmax_reg.softmax_regression import train_softmax_regression
import matplotlib.pyplot as plt

# data process
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = tf.cast(x_train, tf.float32) / 255
x_test = tf.cast(x_test, tf.float32) / 255

batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(batch_size)

# model parameters
num_inputs = 784
num_outputs = 10

# customer size
num_hidden = 256

# 注意w与b之间的关系
w1 = tf.Variable(tf.random.normal(shape=[num_inputs, num_hidden], stddev=0.1, dtype=tf.float32))
b1 = tf.Variable(tf.zeros(shape=num_hidden, dtype=tf.float32))
w2 = tf.Variable(tf.random.normal(shape=[num_hidden, num_outputs], stddev=0.1, dtype=tf.float32))
b2 = tf.Variable(tf.zeros(shape=num_outputs, dtype=tf.float32))


# activation function
def relu(x):
    return tf.maximum(x, 0)


# model definition
def net(x):
    # 分成两步进行计算，
    # 出现的错误：将softmax层写成了relu层，造成结果无法识别
    x = tf.reshape(x, shape=[-1, num_inputs])
    hiden = relu(tf.matmul(x, w1) + b1)
    return tf.math.softmax(tf.matmul(hiden, w2) + b2)


# loss function
def loss_func(pred_y, y):
    return tf.losses.sparse_categorical_crossentropy(y, pred_y)


# model training
num_epoch = 10
lr = 0.5
params = [w1, b1, w2, b2]

# optimizer definition
optimizer = tf.optimizers.SGD(lr)

for epoch in range(num_epoch):
    train_loss, train_acc, test_acc_count = 0.0, 0.0, 0

    for (x, y) in train_iter:
        with tf.GradientTape() as t:
            pred_y = net(x)
            l = tf.reduce_sum(loss_func(pred_y, y))
        grads = t.gradient(l, params)
        # 应用梯度自减
        optimizer.apply_gradients(zip([grad / batch_size for grad in grads], params))

        train_loss += l.numpy()
        train_acc = np.mean(
            tf.cast(tf.argmax(pred_y, axis=1), dtype=tf.int32) == tf.cast(y, dtype=tf.int32))

    for (x, y) in test_iter:
        # 使用 tf.reduce_sum 会报错
        # test_acc_count = tf.reduce_sum(tf.cast(tf.argmax(net(x), axis=1), dtype=tf.int64) == tf.cast(y, tf.int64))
        test_acc_count += np.sum(tf.cast(tf.argmax(net(x), axis=1), dtype=tf.int32) == tf.cast(y, tf.int32))

    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (
        epoch + 1, train_loss / x_train.shape[0], train_acc, test_acc_count / x_test.shape[0]))


# 可视化预测结果（不太重要）
X, y = iter(test_iter).next()


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(tf.reshape(img, shape=(28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(tf.argmax(net(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
