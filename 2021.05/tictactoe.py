#!usr/bin/python3

'''
The idea of this project is to use a NN
as the evaluation function for tic tac toe.

The model should output a 'probability of winning'
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

from itertools import cycle, product
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
        return self._model.predict(np.c_[board.ravel(), whosMove])


class MiniMaxAI:
    def __init__(self, num, evaluationFunction, maxDepth=5):
        self.num = num
        self._f = evaluationFunction
        self.maxDepth = maxDepth

    def minimax(self, board, p, depth, alpha, beta) -> Tuple[int, Optional[int]]:
        maximize = (p == 0)
        if Game.isWinner(board, 1-p):
            return (1 if p else 0) * INF, None
        else if depth >= self.maxDepth:
            return self._f(board, p)
        else:
            bestWeight = -INF if maximize else INF
            bestMove = None
            for move in range(9):
                r, c = divmod(move, 3)
                if not board[r][c].any():
                    board[r][c][p] = 1
                    weight, _ = self.minimax(board, 1-p, depth+1, alpha, beta)
                    board[r][c][p] = 0
                    if maximize:
                        if weight > bestWeight:
                            bestWeight = weight
                            bestMove = move
                        alpha = max(alpha, bestWeight)
                    else:
                        if weight < bestWeight:
                            bestWeight = weight
                            bestMove = move
                        beta = min(beta, bestWeight)
                    if beta <= alpha:
                        break
            if bestMove is None:
                return 0.5, None
            else:
                return bestWeight, bestMove

    def getMove(self, board):
        weight, move = minimax(board, self.num, 0, -INF, INF)
        return move


class Game:
    def __init__(self, playerOne, playerTwo):
        self.board = np.zeros((3, 3, 2))
        self.playerOne = playerOne
        self.playerTwo = playerTwo

    def isWinner(board: np.array, p):
        B = board[:, :, p]
        return any(B[i].all() for i in range(B.shape[0])) or \
            any(B.T[i].all() for i in range(B.shape[1])) or \
            B.diagonal().all() or B.flipud().diagonal().all()

    def play(self):
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
            if isWinner(self.board, p):
                gameHistory['winner'] = p
                break
        return gameHistory


def main():
    evaluationFunction = EvaluationFunction()
    playerOne = MiniMaxAI(0, evaluationFunction)
    playerTwo = MiniMaxAI(1, evaluationFunction)
    game = Game(playerOne, playerTwo)


if __name__ == '__main__':
    main()
