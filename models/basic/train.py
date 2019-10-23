import basic

import tensorflow as tf
from tensorflow import keras

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, y_train ), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = keras.utils.to_categorical(y_train.astype('float32'), 10)
y_test = keras.utils.to_categorical(y_test.astype('float32'), 10)

model = basic.getBasicModel()

model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(x_train, y_train)