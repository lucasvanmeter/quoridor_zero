from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .QuoridorLogic import Board
import numpy as np


class QuoridorGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        b = Board()
        return b.getBoardVec()

    def getBoardSize(self):
        """
        Not used by the NNet in this implementation.
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 81+64+64

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        b = Board(board)
        b.takeAction(player, action)
        return b.getBoardVec(), -1*player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        b = Board(board)
        return b.validActions(player)
    
    def getBestMove(self, board, player):
        b = Board(board)
        return b.bestMove(player)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               Note: these values are from the presective of player so if player 2 negate.
               
        """
        b = Board(board)
        res = b.getWinner(player)
        return res*player

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        b = Board(board)
        if player == 1:
            return b.getBoardVec()
        else:
            return b.getInverseBoardVec()

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        walls = board[4:]
        new_walls = list(np.flipud(np.reshape(walls, (8, 8))).flatten())
        x,y = board[2]
        z,w = board[3]
        new_board = board[0:2] + [(8-x,y),(8-z,w)] + new_walls
        
        pi_1 = pi[0:81]
        pi_2 = pi[81:81+64]
        pi_3 = pi[81+64:]
        new_pi_1 = list(np.flipud(np.reshape(pi_1, (9, 9))).flatten())
        new_pi_2 = list(np.flipud(np.reshape(pi_2, (8, 8))).flatten())
        new_pi_3 = list(np.flipud(np.reshape(pi_3, (8, 8))).flatten())
        new_pi = new_pi_1 + new_pi_2 + new_pi_3
        
        return [(board,pi),(new_board, new_pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board)

    def displayBoard(self, board):
        b = Board(board)
        b.displayBoard()
        
    @staticmethod
    def display(board):
        b = Board(board)
        b.displayBoard()