#
# 文本数据处理，需要较多的预处理步骤，包括中文切词，词典构建，序列填充等
#

import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re,string

train_data_path = '../data/imdb/train.csv'
test_data_path = '../data/imdb/test.csv'

# 考虑最高频的1w词
MAX_WORDS = 10000
# 每个样本保留200个词
MAX_LEN = 200
BATCH_SIZE = 20


# 管道构建
def split_line(line):
    arr = tf.strings.split(line, '\t')
    # 将数字作为标签，并进行行拓展
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    # 将字符作为特征，也是行拓展
    text = tf.expand_dims(arr[1], axis=0)

    return text, label


ds_train_raw = tf.data.TextLineDataset(filenames= [train_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames= [test_data_path]) \
    .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# 构建词典
def clean_text(text):
    lowercase = tf.strings.lower(text)
    # 移除html语句
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')

    # 移除标点符号
    cleaned_punctuation = tf.strings.regex_replace(
        stripped_html, '[%s]' % re.escape(string.punctuation), ''
    )

    return cleaned_punctuation


vectorize_layer = TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS-1,
    output_mode='int',
    output_sequence_length=MAX_LEN
)

ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
# print(vectorize_layer.get_vocabulary()[:100])

# 单词编码
ds_train = ds_train_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test_raw.map(lambda text, label: (vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# 自定义模型构建
class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()

    def build(self, input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size=5, name="conv_1", activation="relu")
        self.pool_1 = layers.MaxPool1D(name="pool_1")
        self.conv_2 = layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.pool_2 = layers.MaxPool1D(name="pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)

    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return (x)

    # 重写summary
    def summary(self):
        x_input = layers.Input(shape=MAX_LEN)
        output = self.call(x_input)
        model = tf.keras.Model(x_input, output)
        model.summary()


model = CnnModel()
model.build(input_shape=(None, MAX_LEN))
# model.summary()


# 模型训练
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


optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        pred_label = model(features, training=True)
        loss = loss_func(labels, pred_label)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 自监督可还行
    train_loss.update_state(loss)
    train_metric.update_state(labels, pred_label)

@tf.function
def valid_step(model, features, labels):
    pred_label = model(features, training=False)
    batch_loss = loss_func(labels, pred_label)

    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, pred_label)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in range(epochs):

        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = 'Epoch={},Loss:{},Accuracy:{},Valid Loss:{},Valid Accuracy:{}'

        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), train_metric.result(), valid_loss.result(),
                                        valid_metric.result())))
            tf.print("")

        # 每轮迭代重新更新记录
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()


train_model(model,ds_train,ds_test,epochs = 6)


# 模型评估
def evaluate_model(model, ds_valid):
    for features, labels in ds_valid:
        valid_step(model, features, labels)
    logs = 'Valid Loss:{},Valid Accuracy:{}'
    tf.print(tf.strings.format(logs, (valid_loss.result(), valid_metric.result())))

    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()


model.save('./save_model/text_model', save_format="tf")