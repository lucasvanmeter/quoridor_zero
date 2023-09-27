"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Modified from origional by

Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import quoridor.QuoridorGame as QG
import quoridor.QuoridorLogic as QL
from random import choice
import random
from collections import namedtuple
import time

class NaiveMCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1, num_rollouts=200):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight
        self.num_rollouts = num_rollouts
        
    def moveAlongShortest(self, node):
        """
        Often the best move is to move along the shortest path.
        
        Returns:
            a pair (board, action) corresponding to the board to move to and the action number that
            takes us there.
        """
        b = QL.Board(node.board)
        short_path = b.getShortestPath(node.player)
        x,y = short_path[1]
        action = 9*x+y
        valid_moves = [i for i,_ in enumerate(b.validActions(node.player)[:81]) if _==1]
        if action not in valid_moves and len(short_path)>2:
            x,y = short_path[2]
            action = 9*x+y
        else:
            action = choice(valid_moves)
        new_node = node.make_move(action)
        return (new_node,action)

    def getActionProb(self, node, temp = 1):
        """
        This function performs n MCTS rollouts starting from node.

        Returns:
            probs: a policy vector (length 81+64+64) where the probability of the ith action is
                   proportional to the average reward
        """
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

#         if node not in self.children:
#             print('Error, node not in children')
        
        for _ in range(self.num_rollouts):
            self.do_rollout(node, heuristic = True)
            
        def score(n):
            if self.N[n] == 0:
                return float(0)  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        kids = node.find_probable_children()
        action_vec = [0]*(81+64+64)
        for board in kids:
            action_vec[kids[board]] = score(board)
        
        best = max(action_vec)
        if best == 0:
            for board in kids:
                action_vec[kids[board]] = 1
                
        prob_sum = sum(action_vec)
        action_vec = [x/prob_sum for x in action_vec]
            
        return action_vec
    
    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        
        best = max(self.children[node], key=score)
        if score(best) == 0:
            return self.moveAlongShortest(node)[0]
        
        return best

    def do_rollout(self, node, heuristic = False):
        "Make the tree one layer better. (Train for one iteration.)"
        start = time.time()
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        if heuristic == True:
            reward = self._heuristic_simulate(leaf)
        else:
            reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_probable_children().keys()

    def _simulate(self, node):
        """
        Returns the reward for a random simulation (to completion) of `node`
        """
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward
            
    def _short_simulate(self,node,n):
        """
        inpute:
            node: current board node
            n: number of random moves before using heuristic reward.
            
        Plays n random moves then returns whichever pawn is closer to it's goal.
        """
        invert_reward = True
        i = 0
        while True and i < n:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward
            i += 1
        return node.heuristic_reward()
    
    def _heuristic_simulate(self,node):
        """
        Simply returns a reward based on which pawn is in a winning position if no walls
        were left.
        """
        return node.heuristic_reward()
        

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return {}

    @abstractmethod
    def find_random_child(self):
        """
        Returns the reward for a random simulation (to completion) of `node`
        """
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True

_QB = namedtuple("QuoridorBoard", "board player")
g = QG.QuoridorGame()

class QuoridorBoard(_QB, Node):
    
#     def __init__(self, board, player):
#         self.board = board
#         self.player = player
        
    @abstractmethod
    def find_children(self):
        """
        All possible successors of this board state. 
        This is stored as a dictoronary of action number:new board
        """
        if g.getGameEnded(self.board, self.player) != 0:
            return {}
        actions = [i for i,v in enumerate(g.getValidMoves(self.board, self.player)) if v==1]
        return {QuoridorBoard(tuple(board),player) : action for (board,player),action in 
                [(g.getNextState(self.board, self.player, action),action) for action in actions]}
    
    def find_probable_children(self):
        """
        Successors of this board state filtered by probable wall placements to cut down on branching. 
        This is stored as a dictoronary of action number:new board
        """
        b = QL.Board(self.board)
        if g.getGameEnded(self.board, self.player) != 0:
            return {}
        actions = [i for i,v in enumerate(b.probableValidActions(self.player)) if v==1]
        return {QuoridorBoard(tuple(board),player) : action for (board,player),action in 
                [(g.getNextState(self.board, self.player, action),action) for action in actions]}
    
    def find_shortest_child(self):
        """
        Finds the child along the shortest path.
        """
        b = QL.Board(self.board)
        short_path = b.getShortestPath(self.player)
        x,y = short_path[1]
        if (self.player == 1 and b.p2pos != (x,y)) or (self.player == -1 and b.p1pos != (x,y)):
            action = 9*x+y
        else:
            if len(short_path)>2:
                x,y = short_path[2]
                action = 9*x+y
            else:
                valid_moves = [i for i,_ in enumerate(b.validActions(self.player)[:81]) if _==1]
                action = choice(valid_moves)
        new_board, new_player = g.getNextState(self.board, self.player, action)
        return QuoridorBoard(tuple(new_board),new_player)
        
    @abstractmethod
    def find_random_child(self):
        """
        Random successor of this board state (for more efficient simulation)
        
        heuristic:
            With a certain probability, move pawn to one of the shortest paths.
            And with the rest probability, half place a wall randomly / half move pawn randomly.
            This heuristic shorten the time taken by rollout phase.
        """
        b = QL.Board(self.board)
        if g.getGameEnded(self.board, self.player) != 0:
            return None
        if random.random() < 0.7:
            return self.find_shortest_child()
        else:
            valid_actions = [i for i,v in enumerate(b.probableValidActions(self.player)) if v==1]
            walls = [i for i in valid_actions if i >= 81]
            moves = [i for i in valid_actions if i < 81]
            if random.random() < 0.5 and len(walls) != 0:
                action = choice(walls)
            else:
                action = choice(moves)
            new_board, new_player = g.getNextState(self.board, self.player, action)
            return QuoridorBoard(tuple(new_board),new_player)
                                        

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        if g.getGameEnded(self.board, self.player) != 0:
            return True
        else:
            return False

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        winner = g.getGameEnded(self.board, 1)
        if winner == 0:
            raise RuntimeError(f"reward called on nonterminal board {self}")
        if winner is self.player:      # It's your turn and you've already won. Should be impossible.
            return 1
        if winner == self.player*-1:
            return 0  # Your opponent has just won. Bad.
        # The winner is neither True, False, nor None
        print(self.player)
        print(self.player*-1)
        print(winner)
        raise RuntimeError(f"board has unknown winner type {board.winner}")
        
    def heuristic_reward(self):
        """
        Plays the game out as if no walls where left. Does not assume the game is over.
        """
        b = QL.Board(self.board)
        winner = b.heuristicWinner(self.player)
        if winner is -1*self.player:      # It's your turn and you've already won. Should be impossible.
            return 1
        if winner == self.player:
            return 0  # Your opponent has just won. Bad.
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {board.winner}")
        
    def to_pretty_string(self):
        g.displayBoard(self.board)
        
    def make_move(self, action):
        new_board, new_player = g.getNextState(self.board, self.player, action)
        return QuoridorBoard(tuple(new_board),new_player)

#     @abstractmethod
#     def __hash__(self):
#         "Nodes must be hashable"
#         return 123456789

#     @abstractmethod
#     def __eq__(node1, node2):
#         "Nodes must be comparable"
#         return True

def new_quoridor_board():
    return QuoridorBoard(board=tuple(g.getInitBoard()), player=1)

def play_game():
    tree = NaiveMCTS()
    board = new_quoridor_board()
    board.to_pretty_string()
    while True:
        action = input("enter action number: ")
        action = int(action)
#         if board.tup[index] is not None:
#             raise RuntimeError("Invalid move")
        board = board.make_move(action)
        board.to_pretty_string()
        if board.is_terminal():
            break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        start = time.time()
        print(board)
        for _ in range(500):
            tree.do_rollout(board, heuristic = True)
        end = time.time()
        print("Time to do rollouts: "+ str(end-start))
        board = tree.choose(board)
        board.to_pretty_string()
        if board.is_terminal():
            break