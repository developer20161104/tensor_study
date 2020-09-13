# 时间序列的预测
# 样例为预测新冠肺炎的时间
#
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses, metrics, callbacks
import os
import datetime

# 可视化原始数据
df = pd.read_csv('../data/covid-19.csv', sep='\t')
# df.plot(x='date', y=['confirmed_num', 'cured_num', 'dead_num'], figsize=[10, 6])
#
# # 旋转下标可还行
# plt.xticks(rotation=60)
#
# plt.show()

# 索引设置（默认索引为数值类型，从0一直渐进），可以设置复合索引，本质类似于查询
dfdata = df.set_index("date")
# 可以当成一张表来使用
dfdiff = dfdata.diff(periods=1).dropna()
# 重置索引
dfdiff = dfdiff.reset_index("date")

# dfdiff.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
# plt.xticks(rotation=60)
dfdiff = dfdiff.drop("date", axis=1).astype("float32")

# 使用前8天预测下一天
WINDOW_SIZE = 8


def batch_dataset(dataset):
    # 表示少于batch_size的情况下删除最后一批
    dataset_batched = dataset.batch(WINDOW_SIZE, drop_remainder=True)
    return dataset_batched


# window is similar to split
ds_data = tf.data.Dataset.from_tensor_slices(tf.constant(dfdiff.values, dtype=tf.float32)) \
    .window(WINDOW_SIZE, shift=1).flat_map(batch_dataset)

ds_label = tf.data.Dataset.from_tensor_slices(
    tf.constant(dfdiff.values[WINDOW_SIZE:], dtype=tf.float32)
)


# cache 使得批次被使用完后继续保存在缓存中，方便后续使用
ds_train = tf.data.Dataset.zip((ds_data, ds_label)).batch(38).cache()


# model create
# 使用函数式API来构建任意的结构模型
class Block(layers.Layer):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

    def call(self, x_input, x):
        x_out = tf.maximum((1+x)*x_input[:, -1, :], 0.0)
        return x_out

    def get_config(self):
        config = super(Block, self).get_config()
        return config


x_input = layers.Input(shape= (None, 3), dtype=tf.float32)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x_input)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x)
x = layers.LSTM(3,return_sequences = True,input_shape=(None,3))(x)
x = layers.LSTM(3,input_shape=(None,3))(x)
x = layers.Dense(3)(x)

x = Block()(x_input,x)
model = models.Model(inputs = [x_input],outputs = [x])
# model.summary()


# 模型训练
# 自定义损失函数，使用的是均方差
class MSPE(losses.Loss):
    def call(self, y_true, y_pred):
        err_percent = (y_true-y_pred)**2 / (tf.maximum(y_true**2, 1e-7))
        mean_err_percent = tf.reduce_mean(err_percent)

        return mean_err_percent

    def get_config(self):
        config = super(MSPE, self).get_config()
        return config


optimizer = tf.keras.optimizers.Adam(lr=0.01)
model.compile(
    optimizer=optimizer,
    loss=MSPE(name='MSPE')
)

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('../data', 'autograph', stamp)

# 回调函数部分值得学习
# tensorboard
tb_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# 如果经过100epoch没有提升，则设置学习率衰减
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=100)
# 如果经过200epoch没有提升，则提前结束训练
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)

callbacks_list = [tb_callback, lr_callback, stop_callback]

history = model.fit(
    # 直接封装使用
    ds_train,
    epochs=500,
    callbacks=callbacks_list
)


# 结果可视化
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()


plot_metric(history,"loss")


# 使用训练好的模型
dfresult = dfdiff[['confirmed_num', 'cured_num','dead_num']].copy()
print(dfresult.tail())


# 预测100天后的走势
for i in range(100):
    input_val = tf.constant(tf.expand_dims(dfresult.values[-38:,:], axis=0))
    # 预测的值需要进行维度转化
    arr_predict = model.predict(input_val)

    dfpredict = pd.DataFrame(
        tf.cast(tf.floor(arr_predict), tf.float32).numpy(),
        columns=dfresult.columns
    )

    dfresult.append(dfpredict, ignore_index=True)

# 结果预测
print(dfresult.query('confirmed_num==0').head())
print(dfresult.query("cured_num==0").head())
print(dfresult.query("dead_num==0").head())

model.save('./save_model/timesort_model', save_format="tf")
print('export saved model.')