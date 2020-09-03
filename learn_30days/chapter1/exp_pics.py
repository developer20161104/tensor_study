#
# 使用cifar10的子集，cifar2预测飞机与汽车两种图像
#

import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import os
import datetime
from tensorboard import notebook
from pathlib import Path

# 创建图像数据的方法
# 1使用tf.keras的ImageDataGenerator来构建生成器
# 2使用tf.data.Dataset 搭配tf.image构建数据管道
# 此处使用方法2
BATCH_SIZE = 100


def load_image(img_path, size=(32, 32)):
    # 使用正则来匹配0,1
    label = tf.constant(1, tf.int8) if tf.strings.regex_full_match(img_path, ".*automobile.*") \
        else tf.constant(0, tf.int8)

    # 对于图像数据，直接使用解码器即可
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size) / 255.0

    return img, label


# 并行化预处理num_parallel_calls 与预存数据 prefetch 提升性能
ds_train = tf.data.Dataset.list_files('../data/cifar2/train/*/*.jpg') \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = tf.data.Dataset.list_files('../data/cifar2/test/*/*.jpg') \
    .map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# plt.figure(figsize=(8, 8))
# for i, (img, label) in enumerate(ds_train.unbatch().take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     ax.imshow(img.numpy())
#     ax.set_title("label = %d" % label)
#     ax.set_xticks([])
#     ax.set_yticks([])
# plt.show()
#
# for x,y in ds_train.take(1):
#     print(x.shape, y.shape)


# 第二种模型构建方式：函数式编程
# 输入维度为特征维度
inputs = layers.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, kernel_size=(3, 3))(inputs)
x = layers.MaxPool2D()(x)
x = layers.Conv2D(64, kernel_size=(5, 5))(x)
x = layers.MaxPool2D()(x)
x = layers.Dropout(rate=0.1)(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

# 时间戳的格式化
# stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir =os.path.join('../data', 'autograph', stamp)
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = str(Path('../data/autograph/' + stamp))

# 使用tensorboard的方法
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(
    optimizer=tf.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    callbacks=[tensorboard_callback],
    workers=4
)


def plot_metric(history, metric):
    plt.figure()
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(history, "loss")
val_loss, val_accuracy = model.evaluate(ds_test, workers=4)
print(val_loss, val_accuracy)

# save
model.save_weights('./save_model/pics_model-weight/model_weights_pics.ckpt', save_format='tf')