"""Testing interpolation of neural net, depending on network architecture."""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
import seaborn as sns
from matplotlib import pyplot as plt

# %% Create synthetic data
SAMPLE_SIZE = 256
X = np.concatenate((np.random.sample(SAMPLE_SIZE) * 0.25,
                np.random.sample(SAMPLE_SIZE) * 0.25 + 0.75))
Y = np.concatenate((np.repeat(0, SAMPLE_SIZE),
                    np.repeat(1, SAMPLE_SIZE)))

sns.scatterplot(x=X, y=Y)
plt.show()


# %% Model
# model architecture
net_cfg = {
    'activation_hidden': 'relu',
}
inputs = Input(shape=(1,))
x = Dense(40, activation=net_cfg['activation_hidden'])(inputs)
x = Dense(40, activation=net_cfg['activation_hidden'])(x)
predictions = Dense(1)(x)

# create model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=Adam(lr=0.01),
              loss='mse',
              metrics=['mae'])

# %% Fit model to data
model.fit(x=X, y=Y, epochs=200, verbose=2)

# %% Predict for new data
x_new = np.random.sample(1000) * 2 - 0.5
y_pred = model.predict(x=x_new).squeeze()

sns.scatterplot(x=x_new, y=y_pred)
plt.show()
