import numpy as np
import tensorflow as tf


# 利用小批量上的均值与标准差，来调节网络中的中间输出
#
# 针对全连层，通常置于仿射变换与激活函数之间
# 针对卷积层，则卷积计算之后，激活函数之前
# 对于预测时，需要进行操作使得单个样本的输出不应该取决于批量归一化需要的随机小批量中的均值与方差

def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not is_training:
        # 预测模式下，直接使用传入的移动平均所得的均值与方差
        x_hat = (x - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            # 全连层
            mean = x.mean(axis=0)
            var = ((x - mean) ** 2).mean(axis=0)
        else:
            # 二维卷积层
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

        # 使用当前的均值与方差来做标准化
        x_hat = (x - mean) / np.sqrt(var + eps)

        # 更新移动平均的均值与方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    # 拉伸与偏移
    y = gamma * x_hat + beta

    return y, moving_mean, moving_var


class BatchNormalization(tf.keras.layers.Layer):
    def __init__(self, decay=0.9, epsilon=1e-5, **kwargs):
        self.decay = decay
        self.epsilon = epsilon

        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=[input_shape[-1], ],
            initializer=tf.initializers.ones,
            trainable=True
        )

        self.beta = self.add_weight(
            name='beta',
            shape=[input_shape[-1], ],
            initializer=tf.initializers.zeros,
            trainable=True
        )

        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[input_shape[-1], ],
            initializer=tf.initializers.zeros,
            trainable=False
        )

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=[input_shape[-1], ],
            initializer=tf.initializers.zeros,
            trainable=False
        )

        super(BatchNormalization, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        # variable = variable * decay + value * (1 - decay)
        delta = variable * self.decay + value * (1 - self.decay)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, list(range(len(inputs.shape) - 1)))
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=variance,
                                           offset=self.beta,
                                           scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


# 由于需要介入中间操作，所以需要将层级拆分成多个部分
net = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=6, kernel_size=5),
     BatchNormalization(),
     tf.keras.layers.Activation('sigmoid'),
     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
     tf.keras.layers.Conv2D(filters=16, kernel_size=5),
     BatchNormalization(),
     tf.keras.layers.Activation('sigmoid'),
     tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(120),
     BatchNormalization(),
     tf.keras.layers.Activation('sigmoid'),
     tf.keras.layers.Dense(84),
     BatchNormalization(),
     tf.keras.layers.Activation('sigmoid'),
     tf.keras.layers.Dense(10, activation='sigmoid')]
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

net.compile(loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=['accuracy'])
history = net.fit(x_train, y_train,
                  batch_size=64,
                  epochs=5,
                  validation_split=0.2)
test_scores = net.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

# 获取归一化层学习到的拉伸参数gamma与偏移参数beta
print(net.get_layer(index=1).gamma, net.get_layer(index=1).beta)
