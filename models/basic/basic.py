import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def getBasicModel():
    inputs = keras.Input(shape=(784,), name='img')
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs)