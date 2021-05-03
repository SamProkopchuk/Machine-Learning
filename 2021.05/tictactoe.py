#!usr/bin/python3

'''
The idea of this project is to use a NN
as the static evaluation function for tic tac toe.

The model should output a real value between -1 and 1
(1 meaning player 1 is winning. -1 meaning player 2 is winning),
given a board position and who's move it is.
(assuming the previous move didn't win the game).

To avoid repeated games, the order in which moves are considered will be randomized.

The architecture:
3x3 nodes for player 1's pieces
3x3 nodes for player 2's pieces
1 node for who's move it is.
'''


import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
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
        return self._model(np.r_[board.ravel(), whosMove][np.newaxis, ...], training=False).numpy().item()


class MiniMaxAI:
    '''
    Assumes player 0 maximizes and player 1 minimizes
    '''

    def __init__(self, num, evaluationFunction, maxDepth=4):
        self.num = num
        self._f = evaluationFunction
        self.maxDepth = maxDepth
        self.moves = np.arange(9)

    def minimax(self, board, p, depth, alpha, beta) -> Tuple[float, Optional[int]]:
        if depth >= self.maxDepth:
            return self._f(board, p), None
        elif board.sum() == 9:
            # Board is full
            return 0, None
        elif p == 0:
            # Is maximizer
            bestWeight = -INF
            bestMove = None
            np.random.shuffle(self.moves)
            for move in self.moves:
                r, c = divmod(move, 3)
                if board[r][c].any():
                    continue
                board[r][c][p] = 1
                if Game.isWinner(board, p):
                    board[r][c][p] = 0
                    return 1, move
                weight, _ = self.minimax(board, 1-p, depth+1, alpha, beta)
                if weight > bestWeight:
                    bestWeight = weight
                    bestMove = move
                board[r][c][p] = 0
                alpha = max(alpha, bestWeight)
                if alpha >= beta:
                    break
            return bestWeight, bestMove
        else:
            # Is minimizer
            bestWeight = INF
            bestMove = None
            np.random.shuffle(self.moves)
            for move in self.moves:
                r, c = divmod(move, 3)
                if board[r][c].any():
                    continue
                board[r][c][p] = 1
                if Game.isWinner(board, p):
                    board[r][c][p] = 0
                    return -1, move
                weight, _ = self.minimax(board, 1-p, depth+1, alpha, beta)
                if weight < bestWeight:
                    bestWeight = weight
                    bestMove = move
                board[r][c][p] = 0
                beta = min(beta, bestWeight)
                if beta <= alpha:
                    break
            return bestWeight, bestMove

    def getMove(self, board):
        _, move = self.minimax(board, self.num, 0, -INF, INF)
        return move


class CLIPlayer:
    def __init__(self, num):
        self.num = num

    def getMove(self, board):
        move = -1
        while not (0 <= move < 9) or board[divmod(move, 3)].any():
            try:
                row, col = map(int, input(
                    'Enter space-separated row and column of tile:').split())
                move = 3 * (row-1) + col-1
            except:
                continue
        return move

class Game:
    def __init__(self, playerOne, playerTwo):
        self.board = np.zeros((3, 3, 2))
        self.playerOne = playerOne
        self.playerTwo = playerTwo

    @staticmethod
    def isWinner(board: np.array, p):
        B = board[:, :, p]
        return any(B[i].all() for i in range(B.shape[0])) or \
            any(B.T[i].all() for i in range(B.shape[1])) or \
            B.diagonal().all() or np.flipud(B).diagonal().all()
        return res

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
        gameHistory = {'winner': None, 'moves': []}
        p = 0
        for _ in range(9):
            if p == 0:
                move = self.playerOne.getMove(self.board)
            else:
                move = self.playerTwo.getMove(self.board)
            row, col = divmod(move, 3)
            self.board[row][col][p] = 1
            gameHistory['moves'].append(move)
            if verbose:
                print(f'Player {p+1} at row {row+1} column {col+1}')
                self.print_board()
            if Game.isWinner(self.board, p):
                gameHistory['winner'] = p
                break
            p = 1-p
        return gameHistory


def main():
    evaluationFunction = EvaluationFunction()
    playerOne = MiniMaxAI(0, evaluationFunction, maxDepth=4)
    playerTwo = CLIPlayer(1)
    game = Game(playerOne, playerTwo)
    game.play(verbose=True)


if __name__ == '__main__':
    main()
