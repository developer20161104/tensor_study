import tensorflow as tf
import tensorflow.keras as keras

# 读取数据
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 归一化处理
x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0

# 定义模型层
class_type = 10

model = keras.Sequential([
    # 使用Flatten层进行压缩（reshape）
    keras.layers.Flatten(input_shape=(28, 28)),
    # 全连层只需要定义一个输出维度
    keras.layers.Dense(class_type, activation=tf.nn.softmax)
])

# 定义softmax与交叉熵损失函数，同时使用防止数值类型的变化带来的不便
loss_func = 'sparse_categorical_crossentropy'

# 定义优化算法
optimizer = keras.optimizers.SGD(lr=0.1)

# 进行模型的训练
# 三要素：优化器，损失函数，评估值
model.compile(
    optimizer=optimizer,
    loss=loss_func,
    metrics=['accuracy']
)

# 也可以在fit中加入校验，作用位置为每一epoch
model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=256,
    validation_data=(x_test, y_test)
)

# 最终模型进行校验
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:', test_acc)
