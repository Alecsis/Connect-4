import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class Connect4Env(object):

    def __init__(self):

        # Board dimension
        self.rows = 6
        self.columns = 7

        # Save the board state
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        # 0: no token
        # 1: agent1 token
        # -1: agent2 token
        self.observation_space = self._new_observation_space()

        # Save the action space
        self.action_space = spaces.Discrete(self.columns)

        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self) -> spaces.Box:
        """
        Reinitialize the environment to the initial state

        :return: state
        """

        #self.obs = np.zeros((self.rows, self.columns), dtype=int)
        self.observation_space = self._new_observation_space()
        return self.observation_space

    def _new_observation_space(self) -> spaces.Box:
        """ Returns the state Box space. """
        # Low bound is excluded
        return spaces.Box(low=-2, high=1,
                          shape=(self.rows, self.columns),
                          dtype=np.int)

    def is_action_valid(self, action: int) -> bool:
        """ If we have a space then the move is valid"""
        try:
            if self.observation_space[0, action] == 0:
                return True
        except:
            pass
        return False

    def draw(self):
        """
        Visualize in the console or graphically the current state
        """
        print("+---" * 7 + '+')
        for row in range(self.rows):
            print(
                '| ' + ' | '.join(list(map(str, list(self.observation_space[row, ::])))) + ' |')
            print("+---" * self.columns + '+')

    def step(self, action: int) -> (spaces.Box, int, bool, dict):
        """
        One can make a step on the environment and obtain its reaction:
        - the new state
        - the reward of the new state
        - should we continue the game?

        :return: state, reward, game_over, info
        """

        # Check if agent's move is valid

        valid_action = self.is_action_valid(action)

        if valid_action:  # Play the move

            i = 1

            while self.observation_space[-i, action] != 0:
                i += 1

            self.observation_space[-i, action] = 1

            # TODO :  -> ADD THE OPPONENT AGENT

            # TODO : COMPUTE THE REWARD AND IF THE GAME IS OVER

            reward = 1 / 42

            done = False

            info = {}

        else:  # End the game and penalize agent
            reward, done, info = -10, True, {}

        return self.obs, reward, done, info


if __name__ == "__main__":
    env = Connect4Env()

    print("[.] Testing Connect4Env Environment")
    print("\t[.] Action space")
    print("Action space:", env.action_space)
    print("Action sample:", env.action_space.sample())
    print("\t[.] Observation space")
    print("Observation space:", env.observation_space)
    print("Observation sample:\n", env.observation_space.sample())
    print("[.] Tests done.")
