import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

batch_size = 600
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(x_train[..., tf.newaxis] / 255, tf.float32),
    tf.cast(y_train, tf.int64)
)).shuffle(1000).batch(batch_size)

test_iter = tf.data.Dataset.from_tensor_slices((
    tf.cast(x_test[..., tf.newaxis] / 255, tf.float32),
    tf.cast(y_test, tf.int64)
)).batch(10000)

# 维度转化为 32-28-28-1 -> 32-26-26-16 -> 32-24-24-16 -> 32-16 -> 32-10
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, input_shape=(28, 28, 1), kernel_initializer=tf.initializers.random_normal),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Conv2D(64, 5),
    tf.keras.layers.MaxPool2D(2, 1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.initializers.random_normal),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=tf.initializers.random_normal)
])

# 模型会自己进行初始化
# for x,y in dataset.take(1):
#     print(x.shape)
#     print("logits: ", model(x).shape)


optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []
for epoch in range(30):
    for (batch, (x, y)) in enumerate(dataset):
        if not batch % 10:
            print('.', end='')

        with tf.GradientTape() as t:
            logits = model1(x, training=True)
            l = loss(y, logits)

        loss_history.append(l.numpy().mean())
        # 关键应用之处
        grads = t.gradient(l, model1.trainable_variables)
        optimizer.apply_gradients(zip(grads, model1.trainable_variables))

    for x, y in test_iter:
        acc = np.mean(tf.cast(tf.argmax(model1(x), axis=1), tf.int64) == y)
    print("epoch {} | acc {:.3%}".format(epoch + 1, acc))
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
# plt.Text(0, 0.5, 'Loss [entropy]')
#
# plt.show()
