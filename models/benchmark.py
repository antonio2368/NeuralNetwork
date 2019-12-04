import timeit

if __name__ == '__main__':

    setup = f'''
from basic import basic
from tensorflow import keras
from tensorflow.keras.models import load_model

import numpy as np

model = load_model('basic/model')

x_test = np.loadtxt('../datasets/mnist-test-input.txt')
    '''

    print(timeit.repeat(stmt='model.predict(x_test)', setup=setup, number=2))
