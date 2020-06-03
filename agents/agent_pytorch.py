from agents.agent import Agent
import torch
import torch.nn as nn
import numpy as np
import tqdm
import random
from gym_connect4.envs.connect4_env import Connect4Env
from gym import Space


class Model(nn.Module):

    def __init__(self, criterion):
        super(Model, self).__init__()
        # The model has 3 inputs (the coordinates of the point) and 3 output (the coordinates of the prediction)
        self.l1 = nn.Linear(42, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, 100)
        self.l4 = nn.Linear(100, 7)
        self.criterion = criterion

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = self.l4(x)
        return x

    def train_on_batch(self, x, y):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.train()
        y_pre = self.__call__(torch.tensor(x, dtype=torch.float))
        loss = self.criterion(torch.tensor(y, dtype=torch.float), y_pre)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class Memory(object):
    def __init__(self, max_memory=1000):
        self.max_memory = max_memory  # maximum elements stored
        self.memory = list()  # initialize the memory

    def size(self):
        return len(self.memory)

    def remember(self, m):
        if len(self.memory) <= self.max_memory:  # if not full
            self.memory.append(m)  # store element m at the end
        else:
            self.memory = self.memory[1:]  # remove the first element
            self.memory.append(m)  # store element m at the end

    def random_access(self, batch_size):
        return random.sample(self.memory, batch_size)  # random sample, from memory, of size batch_size


class AgentTorch(Agent):
    def __init__(self, action_space, observation_space, epsilon=0.1, discount=0.99, batch_size=50):
        super().__init__(action_space, observation_space)
        self.epsilon = epsilon
        self.memory = Memory()
        # Discount for Q learning (gamma)
        self.discount = discount
        self.batch_size = batch_size
        self.model = Model(criterion=nn.MSELoss())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def set_epsilon(self, e):
        self.epsilon = e

    def act(self, s, train=True):
        """ This function should return the next action to do:
        an integer between 0 and 4 (not included) with a random exploration of epsilon"""
        if train:
            if np.random.rand() <= self.epsilon:
                action = self.action_space.sample()
            else:
                action = self.learned_act(s)
        else:  # in some cases, this can improve the performance.. remove it if poor performances
            action = self.learned_act(s)

        return action

    def learned_act(self, s):
        """ Act via the policy of the agent, from a given state s
        it proposes an action a"""
        return torch.argmax(self.model(torch.tensor(s, dtype=torch.float)), axis=1)

    def reinforce(self, s, n_s, a, r, game_over_):
        """ This function is the core of the learning algorithm.
        It takes as an input the current state s_, the next state n_s_
        the action a_ used to move from s_ to n_s_ and the reward r_.

        Its goal is to learn a policy.
        """
        # Two steps: first memorize the states, second learn from the pool

        # 1) memorize
        self.memory.remember([s, n_s, a, r, game_over_])

        # 2) Learn from the pool

        input_states = np.zeros((self.batch_size, 6, 7))
        target_q = np.zeros((self.batch_size, 7))

        if self.memory.size() < self.batch_size:  # if not enough elements in memory we do nothing
            return 1e5  # unknown (loss)

        samples = self.memory.random_access(self.batch_size)

        for i in range(self.batch_size):
            input_states[i], next_s, a, r, end = samples[i]  # state, next_state, action, reward, game_over

            # update the target
            if end:
                target_q[i, a] = r
            else:
                # compute max_a Q(nex_state, a) using the model
                Q_next_state = torch.max(self.model(torch.tensor(next_s.flatten().reshape(1, -1), dtype=torch.float)))

                # r + gamma * max_a Q(nex_state, a)
                target_q[i, a] = r + self.discount * Q_next_state

        # HINT: Clip the target to avoid exploding gradients.. -- clipping is a bit tighter
        target_q = np.clip(target_q, -3, 3)

        # train the model on the batch

        input_data = np.array([input_states[i].flatten().reshape(-1) for i in range(self.batch_size)])
        loss = self.model.train_on_batch(input_data, target_q)

        return loss

    def save(self):
        """ This function returns basic stats if applicable: the
        loss and/or the model"""
        pass

    def load(self):
        """ This function allows to restore a model"""
        pass


def train(agent, env, epoch):
    # Number of won games
    score = 0
    loss = 0

    for e in tqdm.tqdm(range(epoch)):
        # At each epoch, we restart to a fresh game and get the initial state
        state = env.reset()

        # This assumes that the games will terminate
        game_over = False

        win = 0
        lose = 0

        while not game_over:
            # The agent performs an action

            action = agent.learned_act(torch.tensor(state.flatten().reshape(1, -1), dtype=torch.float))
            # take action with the agent

            # Apply an action to the environment, get the next state, the reward
            # and if the games end

            prev_state = state
            state, reward, game_over, info = env.step(action)

            # Update the counters
            if reward > 0:
                win = win + reward
            if reward < 0:
                lose = lose - reward

            # Apply the reinforcement strategy
            loss = agent.reinforce(prev_state, state, action, reward, game_over)

        # Update stats
        score += win - lose

        print(f"Epoch {e}/{epoch}, loss {round(np.float64(loss), 4)}, win/lose count {win}/{lose} ({win - lose})")


if __name__ == '__main__':
    epoch = 1000
    env = Connect4Env()
    agent = AgentTorch(env.action_space, env.observation_space)
    train(agent, env, epoch)
