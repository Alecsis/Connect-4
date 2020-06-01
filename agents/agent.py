import numpy as np
from gym import Space


class Agent(object):

    def __init__(self, action_space: Space, observation_space: Space):
        self.action_space = action_space
        self.observation_space = observation_space

    def get_action(self, obs: np.ndarray) -> int:
        return self.action_space.sample()