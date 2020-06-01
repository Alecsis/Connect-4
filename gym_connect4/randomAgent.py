from gym_connect4.agent import Agent
import random


class RandomAgent(Agent):
    def action(self, board, columns):
        valid_moves = [col for col in range(columns) if board[0, col] == 0]
        return random.choice(valid_moves)
