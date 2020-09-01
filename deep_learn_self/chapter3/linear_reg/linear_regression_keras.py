import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
import numpy as np
import matplotlib.pyplot as plt

# 在内部库中，data模块提供了数据处理的工具
# keras.layers模块定义了神经网络的各种层
# initializers定义了各种初始化方法
# optimizers定义了各种优化算法

# create dataset
num_input = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.3

features = tf.random.normal((num_examples, num_input), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 施加一定噪声
labels += tf.random.normal((num_examples,), stddev=0.01)


# 数据读取
batch_size = 10
# 组合训练数据的标签与特征
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
# 实现小批量读取
dataset = dataset.shuffle(buffer_size=num_examples)
dataset = dataset.batch(batch_size)
dataiter = iter(dataset)

# for (batch, (x,y)) in enumerate(dataset):
#     print(x,y)

# 定义模型
model = keras.Sequential()
# 初始化参数,设置w为正态分布，b为0
model.add(layers.Dense(input_shape=[num_input], units=1, kernel_initializer=init.RandomNormal(stddev=0.01)))
# print(model.get_weights())

# 定义损失函数
loss = tf.losses.MeanSquaredError()

# 定义优化算法
trainer = tf.optimizers.SGD(learning_rate=0.03)

# 模型训练
num_epochs = 4
loss_sort = []

for epoch in range(1, num_epochs+1):
    for (batch, (x, y)) in enumerate(dataset):
        with tf.GradientTape() as t:
            l = loss(model(x, training=True), y)

        # model.trainable_variables 寻找更新的变量
        grads = t.gradient(l, model.trainable_variables)
        # print(model.get_weights())
        # print('\n')
        # print(model.trainable_variables)
        # trainer.apply_gradients 进行参数的更新
        trainer.apply_gradients(zip(grads, model.trainable_variables))

    l = loss(model(features), labels)
    loss_sort.append(l)


plt.plot(tf.range(num_epochs), np.array(loss_sort), '--r')
plt.show()

# 通过model来获取得到的参数
print(model.get_weights())

