from gym import spaces, Env
import numpy as np
from agents.random_agent import RandomAgent


class InvalidAction(Exception):
    pass


class ColumnIsFull(Exception):
    pass


class Connect4Env(Env):

    def __init__(self):

        # Board dimension
        self.nb_rows = 6
        self.nb_columns = 7
        self.done = False

        # nb_empty indicate the number of available space per column
        self.nb_empty = [self.nb_rows] * self.nb_columns

        # Save the board state
        self.state = np.zeros((self.nb_rows, self.nb_columns), dtype=int)

        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.nb_columns)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.nb_rows,
                                                   self.nb_columns),
                                            dtype=np.int)

        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        self.rewards = {
            "invalid": -10,
            "valid":    1/42,
            "won":      1,
            "lost": -1,
            "draw":     0,
        }

        # Render properties
        self.render_tokens = {}
        self.render_tokens[-1] = 'x'
        self.render_tokens[1] = 'o'
        self.render_tokens[0] = ' '

        # Random agent
        self.opponent = RandomAgent(self.action_space, self.state)

        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        """
        Reinitialize the environment to the initial state

        :return: state
        """
        self.state = np.zeros((self.nb_rows, self.nb_columns), dtype=int)
        self.nb_empty = [self.nb_rows] * self.nb_columns
        self.done = False
        return self.state

    def is_action_valid(self, action) -> bool:
        """ If we have a space then the move is valid"""
        return self.nb_empty[action] != 0

    def render(self):
        """
        Visualize in the console or graphically the current state
        """
        print("+---" * self.nb_columns + '+')
        for row in range(self.nb_rows):
            print('| ' + ' | '.join(list(map(lambda x: self.render_tokens[x],
                                             list(self.state[row, ::])))) + ' |')
            print("+---" * self.nb_columns + '+')

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

    def is_boad_full(self) -> bool:
        """
        Check if the board is full
        :return: Bool
        """
        empty_slot_id = 0
        board_full = not (empty_slot_id in self.state)
        return board_full

    def has_player_won(self) -> bool:
        """ Check if the game is over """
        # DIRECTION NORTH EAST
        a1 = [self.state[::-1, :].diagonal(i)
              for i in range(-self.nb_rows + 4, self.nb_columns - 3)]
        # DIRECTION EAST
        a2 = [self.state[i, :] for i in range(self.nb_rows)]
        # DIRECTION SOUTH EST
        a3 = [self.state.diagonal(i)
              for i in range(-self.nb_rows + 4, self.nb_columns - 3)]
        # DIRECTION SOUTH
        a4 = [self.state[:, j] for j in range(self.nb_columns)]

        aligns = a1 + a2 + a3 + a4
        done = any([self.check_line(align)[0] for align in aligns])
        return done

    def emplace_token(self, column, token):
        # Check if column is already full
        if self.nb_empty[column] <= 0:
            raise ColumnIsFull

        # Add the token
        self.nb_empty[column] -= 1
        row = self.nb_empty[column]
        self.state[row, column] = token

    def step(self, action: int) -> (np.ndarray, int, bool, dict):
        """
        One can make a step on the environment and obtain its reaction:
        - the new state
        - the reward of the new state
        - should we continue the game?

        :return: state, reward, game_over, info
        """

        # Init return values
        reward = self.rewards["valid"]
        done = False
        info = {}

        # Let first agent make a move
        try:
            self.emplace_token(action, token=1)
        except ColumnIsFull:
            reward = self.rewards["invalid"]
            done = True
            info = {"Invalid Action"}
            return self.state, reward, done, info

        # Check win condition
        if self.has_player_won():
            reward = self.rewards["won"]
            done = True
            info = {"Won"}
            return self.state, reward, done, info

        # Check draw condition
        if self.is_boad_full():
            reward = self.rewards["draw"]
            done = True
            info = {"Draw"}
            return self.state, reward, done, info

        # Let the opponent agent make a move
        try:
            opp_action = self.opponent.get_action(self.state)
            self.emplace_token(opp_action, token=-1)
        except ColumnIsFull:
            # Opponent mistake, should not happen and we don't want to
            # reward our trained agent for this
            reward = self.rewards["valid"]
            done = True
            info = {"Opponent Invalid Action"}
            return self.state, reward, done, info

        # Check win condition
        if self.has_player_won():
            reward = self.rewards["lost"]
            done = True
            info = {"Lost"}
            return self.state, reward, done, info

        # Check draw condition
        if self.is_boad_full():
            reward = self.rewards["draw"]
            done = True
            info = {"Draw"}
            return self.state, reward, done, info

        self.done = done
        return self.state, reward, done, info


if __name__ == '__main__':
    # Init environment
    env = Connect4Env()
    obs = env.reset()
    done = False

    # Sample actions until game over
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward, done, info)
        env.render()
