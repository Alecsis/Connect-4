from gym import spaces, Env
import numpy as np
from agents.random_agent import RandomAgent


class Connect4Env(Env):

    def __init__(self):
        # Board dimension
        self.rows = 6
        self.columns = 7
        self.done = False

        # nb_empty indicate the number of available space per column
        self.nb_empty = [self.rows] * self.columns

        # Save the board state
        self.obs = np.zeros((self.rows, self.columns), dtype=int)

        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)

        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.rows, self.columns),
                                            dtype=np.int)

        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)

        # Random agent
        self.agent_opp = RandomAgent(self.action_space, self.observation_space)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        """
        Reinitialize the environment to the initial state

        :return: state
        """
        self.obs = np.zeros((self.rows, self.columns), dtype=int)
        self.nb_empty = [self.rows] * self.columns
        self.done = False
        return self.obs

    def set_opponent(self, agent):
        self.agent_opp = agent

    def is_valid(self, action) -> bool:
        """ If we have a space then the move is valid"""
        return self.nb_empty[action] != 0

    def render(self):
        """
        Visualize in the console or graphically the current state
        """
        print("+---" * self.columns + '+')
        for row in range(self.rows):
            print('| ' + ' | '.join(list(map(lambda x: '.' if x == 0 else ('o' if x == 1 else 'x'),
                                             list(self.obs[row, ::])))) + ' |')
            print("+---" * self.columns + '+')

    def check_line(self, align):
        no_token = 0
        token_play1 = 1
        token_play2 = -1
        count_play1, count_play2 = 0, 0
        for token in align:
            if token == no_token:
                count_play1, count_play2 = 0, 0
            elif token == token_play1:
                count_play1 += 1
                count_play2 = 0
            else:
                count_play2 += 1
                count_play1 = 0
            if count_play2 == 4:
                return True, token_play2
            if count_play1 == 4:
                return True, token_play1
        return False, no_token

    def check_draw(self):
        """
        Check if the games ended by a drow
        :return: Bool
        """
        no_token = 0
        if no_token in self.obs:
            return False
        else:
            return True

    def check_over(self):
        """ Check if the game is over """
        # DIRECTION NORTH EAST
        a1 = [self.obs[::-1, :].diagonal(i)
              for i in range(-self.rows + 4, self.columns - 3)]
        # DIRECTION EAST
        a2 = [self.obs[i, :] for i in range(self.rows)]
        # DIRECTION SOUTH EST
        a3 = [self.obs.diagonal(i)
              for i in range(-self.rows + 4, self.columns - 3)]
        # DIRECTION SOUTH
        a4 = [self.obs[:, j] for j in range(self.columns)]

        aligns = a1 + a2 + a3 + a4
        done = any([self.check_line(align)[0] for align in aligns])
        if done:
            token_winner = 1
        else:
            done = self.check_draw()
            token_winner = 0

        return done, token_winner

    def play(self, action, token):
        if self.is_valid(action):
            self.nb_empty[action] -= 1
            rowidx = self.nb_empty[action]
            self.obs[rowidx, action] = token

    def step(self, action: int) -> (np.ndarray, int, bool, dict):
        """
        One can make a step on the environment and obtain its reaction:
        - the new state
        - the reward of the new state
        - should we continue the game?

        :return: state, reward, game_over, info
        """

        # Check if agent's move is valid
        if self.is_valid(action) and not self.done:  # Play the move
            self.play(action, token=1)
            done, token_winner = self.check_over()
            if token_winner == 1:
                reward = 1
            elif done:
                reward = 0
            else:
                reward = 1 / 42
            info = {}
        else:  # End the game and penalize agent
            reward, done, info = -10, True, {}

        if not done:
            action_opp = self.agent_opp.get_action(self.obs)
            self.play(action_opp, token=-1)
            done, token_winner = self.check_over()
            if token_winner == -1:
                reward = -1
            elif done:
                reward = 0

        self.done = done
        return self.obs, reward, done, info


if __name__ == '__main__':
    # Init environment
    env = Connect4Env()
    obs = env.reset()
    done = False

    # Sample actions until game over
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
    env.render()
