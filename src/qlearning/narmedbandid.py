"""Simple neural net for multi-armed bandid."""

import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


epsilon = 0.1  # epsilon of epsilon-greedy strategy


def bandid(arm: int):
    """Retrieve reward for arm of bandid."""
    mean = arm
    sd = 1
    rand = np.random.normal(arm, sd)
    return rand

# create model
model = Sequential()
model.add(Dense(24, input_dim=1, activation="relu"))
model.add(Dense(48, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error",
              optimizer=Adam(lr=0.01))

# predict for all arms
def predict_rewards(model: Sequential):
    """Predict rewards for actions."""
    return [model.predict(np.array([i]))[0][0] for i in range(5)]

def best_arm(model: Sequential):
    """Returns index of best prediction."""
    prediction = predict_rewards(model)
    return prediction.index(max(prediction))

# Probe random arm and learn.
history = []
for i in range(100):
    if random.random() < epsilon:
        # here: choose random arm
        arm = random.choice(range(5))
    else:
        # here: choose arm with best prediction
        arm = best_arm(model)
    reward = bandid(arm)
    model.fit(np.array([arm]), np.array([reward]), epochs=1, verbose=0)
    # save reward
    history.append((arm, reward))

# show history
reward_history = [i[1] for i in history]
arm_history = [i[0] for i in history]

df = pd.DataFrame({'action': np.array(arm_history),
                   'reward': np.array(reward_history)})
df.plot()
