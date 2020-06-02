import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt


ENV_NAME = "gym_connect4:connect4-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        lin = observation_space.shape[0]
        col = observation_space.shape[1]

        # Conv2D model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=(lin, col, 1)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.action_space, activation='softmax'))
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adadelta(
                               learning_rate=LEARNING_RATE),
                           metrics=['accuracy'])

        # Linear model
        # self.model = Sequential()
        # self.model.add(Dense(24, input_shape=(lin*col,), activation="relu"))
        # self.model.add(Dense(24, activation="relu"))
        # self.model.add(Dense(self.action_space, activation="linear"))
        # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # At the beginning, high probability of random play
        # i.o. to build some dataset
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        # then, let the model predict
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        # Stack batch of memory first
        if len(self.memory) < BATCH_SIZE:
            return
        # When the memory is big enough, take a sample from it
        batch = random.sample(self.memory, BATCH_SIZE)
        # For each state/next_state/reward combination:
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                pred_next = self.model.predict(state_next)
                q_update = (reward + GAMMA *
                            np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # Train model
            self.model.fit(state, q_values, verbose=0)
        # Decay exploration to allow model to make some predictions
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def main():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space
    lin = observation_space.shape[0]
    col = observation_space.shape[1]
    logs = []
    rewards = []
    action_space = env.action_space.n
    dqn_solver = DQNSolver(env.observation_space, action_space)
    run = 0
    try:
        while True:
            run += 1
            state = env.reset()
            # state = np.reshape(state, (1, lin * col)) # Linear model
            state = np.reshape(state, (1, lin, col, 1))  # Conv2D model
            step = 0
            while True:
                step += 1
                action = dqn_solver.act(state)
                state_next, reward, terminal, info = env.step(action)
                # state_next = np.reshape(state_next, (1, lin * col)) # Linear model
                state_next = np.reshape(
                    state_next, (1, lin, col, 1))  # Conv2D model
                dqn_solver.remember(state, action, reward,
                                    state_next, terminal)
                state = state_next
                if terminal:
                    print("Run: " + str(run) + ", exploration: " +
                          str(dqn_solver.exploration_rate) + ", score: " + str(step))
                    print("Reward: ", reward)
                    print(info)
                    rewards.append(max(reward, -1))
                    logs.append("Run: {}, exploration: {}, {}, {}\n".format(
                        str(run), str(dqn_solver.exploration_rate), reward, info))
                    env.render()
                    break
                dqn_solver.experience_replay()
    except KeyboardInterrupt:
        with open("alexis/logs.txt", "w+") as f:
            f.writelines(logs)
        plt.title("Rewards evolution")
        plt.xlabel("Run")
        plt.ylabel("Reward")
        plt.plot(rewards)
        plt.show()


if __name__ == "__main__":
    main()
