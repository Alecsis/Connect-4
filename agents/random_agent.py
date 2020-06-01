from agents.agent import Agent
import random
import numpy as np


class RandomAgent(Agent):
    def get_action(self, obs):
        return random.choice(self.get_valid_actions(obs))

    def get_valid_actions(self, obs: np.ndarray) -> list:
        valid_actions = [col for col in range(obs.shape[1]) if obs[0, col] == 0]
        return valid_actions