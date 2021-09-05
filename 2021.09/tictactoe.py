#!usr/bin/python3

'''
The idea of this project is to use a NN
as the static evaluation function for tic tac toe.

The model should output a real value between -1 and 1
(1 meaning player 1 is winning. -1 meaning player 2 is winning),
given a board position and who's move it is.
(assuming the previous move didn't win the game).

The architecture:
3x3 nodes for player 1's pieces
3x3 nodes for player 2's pieces
1 node for who's move it is.
'''
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, InputLayer, Sequential
from typing import Optional, Tuple


MAXDEPTH = 10
INF = 1 << 10


class EvaluationFunction:
    def __init__(self):
        self._model = EvaluationFunction._get_model()
        self._model.compile(
            optimizer='SGD',
            loss='binary_crossentropy',
            metrics=['accuracy'])

    def _get_model():
        res = Sequential()
        res.add(InputLayer(input_shape=19))
        res.add(Dense(16, activation='relu'))
        res.add(Dropout(0.2))
        res.add(Dense(8, activation='relu'))
        res.add(Dense(1, activation='tanh'))
        return res

    def fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)

    def __call__(self, board, whosMove):
        return self._model(np.r_[board.ravel(), whosMove]
                           [np.newaxis, ...], training=False).numpy().item()


class MiniMaxAI:
    '''
    Assumes player 0 maximizes and player 1 minimizes
    '''

    def __init__(self, num, evaluation_function, max_depth=4):
        self.num = num
        self._f = evaluation_function
        self.max_depth = max_depth

    def minimax(self, board, p, depth, alpha,
                beta) -> Tuple[float, Optional[int]]:
        if Game.isWinner(board, 1 - p):
            return (1 if (1 - p == 0) else -1), None
        elif board.sum() == 9:
            # Board is full
            return 0, None
        elif depth >= self.max_depth:
            return self._f(board, p), None
        else:
            best_weight = INF if p else -INF
            best_move = None
            # Use np.random.permutation to not select same moves each time lol
            for move in np.random.permutation(9):
                r, c = divmod(move, 3)
                if board[r][c].any():
                    continue
                board[r][c][p] = 1
                weight, _ = self.minimax(board, 1 - p, depth + 1, alpha, beta)
                board[r][c][p] = 0
                if p:
                    if weight < best_weight:
                        best_weight = weight
                        best_move = move
                    beta = min(beta, best_weight)
                else:
                    if weight > best_weight:
                        best_weight = weight
                        best_move = move
                    alpha = max(alpha, best_weight)
                if alpha >= beta:
                    break
            return best_weight, best_move

    def get_move(self, board):
        weight, move = self.minimax(board, self.num, 0, -INF, INF)
        # print(f'Current evaluation is {weight}')
        return move


class CLIPlayer:
    def __init__(self, num):
        self.num = num

    def get_move(self, board):
        move = -1
        while not (0 <= move < 9) or board[divmod(move, 3)].any():
            try:
                row, col = map(int, input(
                    'Enter space-separated row and column of tile:').split())
                move = 3 * (row - 1) + col - 1
            except BaseException:
                continue
        return move


class Game:
    def __init__(self, player1, player2):
        self.board = np.zeros((3, 3, 2))
        self.player1 = player1
        self.player2 = player2

    @staticmethod
    def isWinner(board: np.array, p):
        B = board[:, :, p]
        return any(B[i].all() for i in range(B.shape[0])) or \
            any(B.T[i].all() for i in range(B.shape[1])) or \
            B.diagonal().all() or np.flipud(B).diagonal().all()

    def print_board(self):
        for r in range(3):
            for c in range(3):
                if self.board[r][c][0]:
                    print('|X', end='')
                elif self.board[r][c][1]:
                    print('|O', end='')
                else:
                    print('| ', end='')
            print('|')

    def play(self, verbose=False):
        game_history = {'winner': None, 'moves': []}
        p = 0
        for _ in range(9):
            if p == 0:
                move = self.player1.get_move(self.board)
            else:
                move = self.player2.get_move(self.board)
            row, col = divmod(move, 3)
            self.board[row][col][p] = 1
            game_history['moves'].append(move)
            if verbose:
                print(f'Player {p+1} at row {row+1} column {col+1}')
                self.print_board()
            if Game.isWinner(self.board, p):
                game_history['winner'] = p
                break
            p = 1 - p
        return game_history


def main():
    evaluation_function = EvaluationFunction()
    player1 = MiniMaxAI(0, evaluation_function, max_depth=5)
    player2 = MiniMaxAI(1, evaluation_function, max_depth=5)
    game_histories = []
    for gameno in range(100):
        game = Game(player1, player2)
        game_history = game.play()
        print(f'Game result: {game_history}')
        game_histories.append(game_history)
    df = pd.DataFrame(game_histories)
    pathlib.Path('./data').mkdir(exist_ok=True)
    df.to_csv('./data/histories.csv', index=False)


if __name__ == '__main__':
    main()
