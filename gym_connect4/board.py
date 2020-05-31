import numpy as np
from gym import spaces


class ConnectFour(object):

    def __init__(self):

        self.rows = 6
        self.columns = 7

        # Save the board state
        self.obs = np.zeros((self.rows, self.columns), dtype=int)

        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.rows, self.columns),
                                            dtype=np.int)

        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):

        """
        Reinitialize the environment to the initial state

        :return: state
        """

        self.obs = np.zeros((self.rows, self.columns), dtype=int)

    def valid(self, action) -> bool:
        """ If we have a space then the move is valid"""
        if self.obs[0, action] == 0:
            return True
        return False

    def draw(self):
        """
        Visualize in the console or graphically the current state
        """
        print("+---" * 7 + '+')
        for row in range(self.rows):
            print('| ' + ' | '.join(list(map(str, list(self.obs[row, ::])))) + ' |')
            print("+---" * self.columns + '+')

    def step(self, action):
        """
        One can make a step on the environment and obtain its reaction:
        - the new state
        - the reward of the new state
        - should we continue the game?

        :return: state, reward, game_over, info
        """

        # Check if agent's move is valid

        is_valid = self.valid(action)

        if is_valid:  # Play the move

            i = 1

            while self.obs[-i, action] != 0:
                i += 1

            self.obs[-i, action] = 1

            reward = 1 / 42

            done = False

            info = {}

        else:  # End the game and penalize agent
            reward, done, info = -10, True, {}

        return self.obs, reward, done, info


