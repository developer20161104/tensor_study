import tensorflow as tf
import numpy as np

# gpu setting
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# model structure of AlexNet
net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

x = tf.random.uniform((1, 224, 224, 1))

# print output shape
for layer in net.layers:
    x = layer(x)
    # print(layer.name, 'output_shape\t', x.shape)


class DataLoader:
    # preliminary
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float)/255.0, axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float)/255.0, axis=-1)

        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)

        self.num_train = self.train_labels.shape[0]
        self.num_test = self.test_labels.shape[0]

    def get_batch_train(self, batch_size):
        # index of the numpy is a handy way to converge the discrete position
        index = np.random.randint(0, self.num_train, batch_size)

        # resize to 224, 224
        resized_image = tf.image.resize_with_pad(self.train_images[index], 224, 224, )
        return resized_image.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, self.num_test, batch_size)

        # resize to 224, 224
        resized_image = tf.image.resize_with_pad(self.test_images[index], 224, 224, )
        return resized_image.numpy(), self.test_labels[index]


batch_size = 128
dataloader = DataLoader()
# x_batch, y_batch = dataloader.get_batch_train(batch_size)
# print('x_batch_shape: ',x_batch.shape, ' y_batch_shape: ', y_batch.shape)


def train_alexnet():
    epoch = 5
    num_iter = dataloader.num_train // batch_size

    for e in range(epoch):
        for n in range(num_iter):

            x_batch, y_batch = dataloader.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)

            if not n%20:
                net.save_weights('5.6_alexnet_weights.h5')


optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, nesterov=False)

net.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

x_batch, y_batch = dataloader.get_batch_train(batch_size)
net.fit(x_batch, y_batch)

# train_alexnet()