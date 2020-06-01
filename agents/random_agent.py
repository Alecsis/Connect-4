from agents.agent import Agent
import random
import numpy as np


class RandomAgent(Agent):
    def action(self, obs):
        valid_moves = [col for col in range(obs.shape[1]) if obs[0, col] == 0]
        return random.choice(valid_moves)