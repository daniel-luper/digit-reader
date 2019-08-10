import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist = input_data.read_data_sets("MNIST/", one_hot=False)
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels

# change format
x_train = np.reshape(x_train, (-1, 28, 28))
x_test = np.reshape(x_test, (-1, 28, 28))

# normalize
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
