"""Simple perceptron for understanding Keras better."""

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import os
import random

# suppress tensorflow installation warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define model
model = Sequential()
model.add(Dense(1, input_dim=2, kernel_initializer='normal', use_bias=False))
model.compile(loss='mean_absolute_error', optimizer=SGD(lr=0.01))

# set initial weights
weights = model.get_weights()
weights[0] = np.array([[.5], [.5]])
model.set_weights(weights=weights)

# check initial weights
model.get_weights()[0].tolist()

# train with 1 observation each
for i in range(0, 1000):
    # print weights
    print(model.get_weights()[0].tolist())
    # train on one example
    x = np.array([[1, random.random()]])
    model.fit(x=x, y=np.array([1]), batch_size=1, epochs=1, verbose=0)
