
from tensorflow.python import keras as k
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizers import Adam
import numpy as np

# %% DESIGN NETWORK

DROPOUT = 0.1
LAYER_WIDTH = 200

inputs = k.layers.Input(shape=(1,))
l1 = k.layers.Dense(LAYER_WIDTH, activation='sigmoid')(inputs)
l2 = k.layers.Dropout(DROPOUT)(l1, training=True)
l3 = k.layers.Dense(LAYER_WIDTH, activation='sigmoid')(l2)
l4 = k.layers.Dropout(DROPOUT)(l3, training=True)
l5 = k.layers.Dense(LAYER_WIDTH, activation='sigmoid')(l4)
l6 = k.layers.Dropout(DROPOUT)(l5, training=True)
output = k.layers.Dense(1)(l6)

model = k.Model(inputs, output)
model.compile(loss=MSE, optimizer=Adam(lr=0.001))

# %% FIT AND EVALUATE

model.fit(x=np.array([[0], [1]]),
          y=np.array([0, 1]),
          verbose=2, epochs=10000, batch_size=10)


# %% PREDICT
def predict(model_, x_):
    a = np.empty(1000)
    a.fill(x_)
    mean_ = model_.predict(a).mean()
    std_ = model_.predict(a).std()
    return mean_, std_


for x in [0, 0.25, 0.5, 0.75, 1]:
    mean, std = predict(model, x)
    print(f"prediction for {x}: mean: {mean}; std: {std}")
