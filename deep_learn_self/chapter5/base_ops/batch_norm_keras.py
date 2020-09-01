import tensorflow as tf

net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Conv2D(filters=6, kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Conv2D(filters=16, kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(120))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(84))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(10, activation='sigmoid'))

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
