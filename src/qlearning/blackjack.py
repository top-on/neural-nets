
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=100)
        
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.tau = .1

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=3, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(2))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.sample([0, 1], 1)[0]
        else:
            return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size: int):
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        # update epsilon
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('Blackjack-v0')
    agent = DQNAgent(env)
    history_100 = deque(maxlen=100)
    history_1000 = deque(maxlen=1000)
    # Play the game many episodes
    for episode in range(100000):
        wins = 0.0
        losses = 0.0
        draws = 0.0
        reward_sum = 0.0
        for tournament in range(100):
            # reset state in the beginning of each game
            state = env.reset()
            state = np.reshape(state, [1,3])
            # Our goal is to beat the dealer in blackjack
            while True:
                # Choose action (epsilon-greedy strategy)
                action = agent.act(state)
                # Advance the game to the next state.
                # Reward is 1 for winning, 0 for draw or ongoing, -1 for loss
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 3])
                # Remember the previous state, action, reward, and done
                agent.remember(state, action, reward, next_state, done)
                # make next_state the new current state for the next frame
                state = next_state
                # done becomes True when the game ends
                if done:
                    reward_sum += reward
                    # increment the event counters
                    if reward == 1.0:
                        wins += 1
                    if reward == -1.0:
                        losses += 1
                    if reward == 0:
                        draws += 1
                    # break out of the loop
                    break
        # train the agent after each tournament of 100 games
        agent.replay(100)
        agent.target_train()
        # save reward sum
        history_100.append(reward_sum)
        history_1000.append(reward_sum)


        # print the success
        print("episode: {}, wins: {}, losses: {}, draws: {}, reward_sum: {}, eps.: {}, reward_100: {}, reward_1000: {}"
              .format(episode, wins, losses, draws, reward_sum, agent.epsilon, np.mean(history_100), np.mean(history_1000)))
