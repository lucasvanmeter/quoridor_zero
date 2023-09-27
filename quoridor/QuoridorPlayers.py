import numpy as np
import ast

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanQuoridorPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
#         for i in range(len(valid)):
#             if valid[i]:
#                 print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_type = input("enter move or wall: ")
            if input_type == "move":
                move = input("enter move ((0,0) is left, top) :")
                x,y = ast.literal_eval(move)
                action = x*9+y
            elif input_type == "wall":
                wall = input("enter wall position and orientation ((0,0,1) is left, top, vertical) :")
                x,y, orientation = ast.literal_eval(wall)
                if orientation == 0:
                    action = 81+x*8+y
                if orientation == 1:
                    action = 81+64+x*8+y
            else:
                print("Not recognized!")
            if valid[action]:
                    break
            else:
                print("That is not a valid action.")
        return action


# class GreedyOthelloPlayer():
#     def __init__(self, game):
#         self.game = game

#     def play(self, board):
#         valids = self.game.getValidMoves(board, 1)
#         candidates = []
#         for a in range(self.game.getActionSize()):
#             if valids[a]==0:
#                 continue
#             nextBoard, _ = self.game.getNextState(board, 1, a)
#             score = self.game.getScore(nextBoard, 1)
#             candidates += [(-score, a)]
#         candidates.sort()
#         return candidates[0][1]
