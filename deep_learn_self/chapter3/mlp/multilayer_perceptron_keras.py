import tensorflow as tf
import tensorflow.keras as keras

fashion_mnist = keras.datasets.fashion_mnist


# model definition
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255
x_test = tf.cast(x_test, tf.float32) / 255

model.compile(
    optimizer= keras.optimizers.SGD(lr=0.5),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(x_train, y_train,
          batch_size= 256,
          epochs= 10,
          validation_data=(x_test, y_test),
          validation_freq=1)
