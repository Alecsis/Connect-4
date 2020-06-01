from gym_connect4.agent import Agent
import random
import numpy as np


class RandomAgent(Agent):
    def action(self, board, columns):
        valid_moves = [col for col in range(columns) if board[0, col] == 0]
        return random.choice(valid_moves)

class OneStepAgent(Agent):
    def __init__(self, rows, columns, token):
        self.rows = rows
        self.columns = columns
        self.token = token

    def action(self, board, columns, token = -1):
        valid_moves = [col for col in range(columns) if board[0, col] == 0]
        best_move = valid_moves[0]
        record = -1e7
        for move in valid_moves:
            next_board = board.copy()
            i = 0
            while next_board[i, move] == 0:
                i += 1
            i -= 1
            next_board[i, move] = token
            score_p = self.evaluate_heuristic(self, next_board)
            next_board[i, move] = - token
            score_n = self.evaluate_heuristic(self, next_board)
            total = score_p - score_n
            if total > record:
                best_move = move
                record = total
        return best_move

    def evaluate(self, align, token):
        score = 0
        adv_token = - token
        for i in range(len(align)-4):
            window = align[i:i+4]
            if token and adv_token in window: continue
            if token in window:
                sign = 1
                score = max(score, abs(sum(window)))
            if adv_token in window:
                sign = -1
                score = min(score, -abs(sum(window)))
        return sign*(10**score)


    def evaluate_heuristic(self, board):
        score = 0
        a1 = [board[::-1, :].diagonal(i) for i in range(-self.rows + 4, self.columns - 3)]

        # DIRECTION EAST
        a2 = [board[i, :] for i in range(self.rows)]

        # DIRECTION SOUTH EST
        a3 = [board.diagonal(i) for i in range(-self.rows + 4, self.columns - 3)]

        # DIRECTION SOUTH
        a4 = [board[:, j] for j in range(self.columns)]

        aligns = a1 + a2 + a3 + a4

        for align in aligns:
            score += self.evaluate(align, self.token)

        return score
