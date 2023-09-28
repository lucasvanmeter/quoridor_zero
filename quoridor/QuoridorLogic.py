import copy
import numpy as np
import time

class Board():
    """
    The board can be generate from the current board positions stored as a vector.
    This vector has length 2 + 2 + 64, corresponding to 
    (a) the players remaining walls, [p1,p2]
    (b) the position of p1 pawn and p2 pawn [(a,b),(x,y)]
    (c) the wall spots with 0 if no wall, 1 if horizontal wall, and 2 if vertical wall.
    An initial board vector is given by [10,10]+[(4,0),(4,8)]+64*[0]
    
    We use this generation technique to play nicely with existing code in which the board is passed
    around as just this vector. We can also pass the NNet this vector as input.
    
    It is also easy to get a canonical board position by inverting the board and switching the positions
    of p1 and p2 pawns.
    
    It is always assumed that player 1 is currently going.
    """
    def __init__(self, vec = [10,10]+[(4,0),(4,8)]+64*[0]):
        self.vec = vec
        self.p1walls = vec[0]
        self.p2walls = vec[1]
        self.p1pos = vec[2]
        self.p2pos = vec[3]
        
        # Walls are stored in an 8x8 grid which can be accesed by [x,y] coordinates.
        walls = vec[4:]
        self.walls = np.array([walls[8*i:8+8*i] for i in range(8)])
        
        # The board is stored as graph. The data structure is a dictionary with the keys as
        # positions and the values and adjacent connected positions.
        graph = {}
        for x in range(9):
            for y in range(9):
                graph[(x,y)] = []
                if x > 0:
                    graph[(x,y)].append((x-1,y))
                if x < 8:
                    graph[(x,y)].append((x+1,y))
                if y > 0:
                    graph[(x,y)].append((x,y-1))
                if y < 8:
                    graph[(x,y)].append((x,y+1))
        self.boardGraph = graph
        
        for x in range(8):
            for y in range(8):
                if self.walls[x,y] == 1:
                    self.boardGraph[(x,y)].remove((x,y+1))
                    self.boardGraph[(x,y+1)].remove((x,y))
                    self.boardGraph[(x+1,y)].remove((x+1,y+1))
                    self.boardGraph[(x+1,y+1)].remove((x+1,y))
                if self.walls[x,y] == 2:
                    self.boardGraph[(x,y)].remove((x+1,y))
                    self.boardGraph[(x+1,y)].remove((x,y))
                    self.boardGraph[(x,y+1)].remove((x+1,y+1))
                    self.boardGraph[(x+1,y+1)].remove((x,y+1))
        
    def getBoardVec(self):
        """
        Returns: A vector of length 2+2+64 encoding the remaining walls, squares, and walls.
        """
        return [self.p1walls, self.p2walls, self.p1pos, self.p2pos] + list(self.walls.flatten())
    
    def getInverseBoardVec(self):
        remWalls = [self.p2walls, self.p1walls]
        pos = [(self.p2pos[0],8-self.p2pos[1]),(self.p1pos[0],8-self.p1pos[1])]
        walls = [x for col in self.walls for x in col[::-1]]
        return remWalls + pos + walls

    
    def validPawnMoves(self, player):
        """
        Input:
            player: 1 or -1
        
        Description:    
            Normall you can move to any adjacent square not blocked by a wall. If your 
            opponents pawn occupies an adjacent square you can jump over them. If there is
            a wall behind them then you can jump and turn over them.

        Returns:
            A list of all valid squares the player may move to.
        """
        if player == 1:
            x,y = self.p1pos
            other = self.p2pos
        else:
            x,y = self.p2pos
            other = self.p1pos
        
        moves = []
        # Check left and right
        for i in [1,-1]:
            if (x+i,y) in self.boardGraph[(x,y)]:
                # if space open add to moves
                if other != (x+i,y):
                    moves.append((x+i,y))
                else: 
                    # otherwise see if jumping over is allowed
                    if (x+2*i,y) in self.boardGraph[(x+i,y)]:
                        moves.append((x+2*i,y))
                    else:
                        # last, add the diagonal jumps if available.
                        if (x+i,y+i) in self.boardGraph[(x+i,y)]:
                            moves.append((x+i,y+i))
                        if (x+i,y-i) in self.boardGraph[(x+i,y)]:
                            moves.append((x+i,y-i))
        
        for i in [1,-1]:
            if (x,y+i) in self.boardGraph[(x,y)]:
                if other != (x,y+i):
                    moves.append((x,y+i))
                else: 
                    if (x,y+2*i) in self.boardGraph[(x,y+i)]:
                        moves.append((x,y+2*i))
                    else:
                        if (x+i,y+i) in self.boardGraph[(x,y+i)]:
                            moves.append((x+i,y+i))
                        if (x-i,y+i) in self.boardGraph[(x,y+i)]:
                            moves.append((x-i,y+i))
                            
        return moves
    
    def movePawn(self, x,y, player):
        """
        x,y: coordinates to move the pawn to.
        player: 1 or -1
        
        Updates self.p1pos or self.p2pos
        """
        if player == 1:
            self.p1pos = (x,y)
        else:
            self.p2pos = (x,y)
        
    def validWalls(self, player):
        """
        player: 1 or -1
        
        Returns a length 64+64 list of 1's and 0's. In particular a wall cannot 
        be placed if it would hit another wall, or if it would cut of a player from their goal. 
        This second case (which requires a DFS of the board graph) is handeled by testWallPlacement()
        """
        if player == 1 and self.p1walls == 0:
            return 128*[0]
        if player == -1 and self.p2walls == 0:
            return 128*[0]
        
        vert = 64*[0]
        horz = 64*[0]
        for x in range(8):
            for y in range(8):
                n = 8*x+y
                if self.walls[x,y] == 0:
                    if x == 0:
                        if self.walls[x+1,y] != 1:
                            if self.testWallPlacement(x,y,1):
                                horz[n]=1
                    elif x == 7:
                        if self.walls[x-1,y] != 1:
                            if self.testWallPlacement(x,y,1):
                                horz[n]=1
                    elif self.walls[x+1,y] != 1 and self.walls[x-1,y] != 1:
                        if self.testWallPlacement(x,y,1):
                            horz[n]=1

                    if y == 0:
                        if self.walls[x,y+1] != 2:
                            if self.testWallPlacement(x,y,2):
                                vert[n]=1
                    elif y == 7:
                        if self.walls[x,y-1] != 2:
                            if self.testWallPlacement(x,y,2):
                                vert[n]=1
                    elif self.walls[x,y+1] != 2 and self.walls[x,y-1] != 2:
                        if self.testWallPlacement(x,y,2):
                            vert[n]=1
                            
        return horz+vert
    
    def probableWalls(self):
        """
        returns a list of length 64+64 corresponding to wall placements that are:
            touching a pawn
            touching a previously placed wall
            horizontal walls touching the edges
        """
        #Touching previous walls or horizotal walls touching edges
        horz = 64*[0]
        for x in range(8):
            for y in range(8):
                n = 8*x+y
                touch_left = (x == 0) or \
                        (x > 0 and (self.walls[x-1,y] == 2) or \
                                    (y > 0 and self.walls[x-1,y-1] == 2) or \
                                    (y < 7 and self.walls[x-1,y+1] == 2)) or \
                        (x > 1 and self.walls[x-2,y] == 1)
            
                touch_mid = (y > 0 and (self.walls[x,y-1] == 2)) or \
                            (y < 7 and (self.walls[x,y+1] == 2))

                touch_right = (x == 7) or \
                            (x < 7 and (self.walls[x+1,y] == 2) or \
                                        (y > 0 and self.walls[x+1,y-1] == 2) or \
                                        (y < 7 and self.walls[x+1,y+1] == 2)) or \
                            (x < 6 and self.walls[x+2,y] == 1)
                if touch_left or touch_mid or touch_right:
                    horz[n] = 1

        vert = 64*[0]
        for x in range(8):
            for y in range(8):
                n = 8*x+y
                touch_up = (y > 0 and (self.walls[x,y-1] == 1) or \
                                    (x > 0 and self.walls[x-1,y-1] == 1) or \
                                    (x < 7 and self.walls[x+1,y-1] == 1)) or \
                        (y > 1 and self.walls[x,y-2] == 2)
            
                touch_mid = (x > 0 and (self.walls[x-1,y] == 1)) or \
                            (x < 7 and (self.walls[x+1,y] == 1))

                touch_down = y < 7 and ((self.walls[x,y+1] == 1) or \
                                        (x > 0 and self.walls[x-1,y+1] == 1) or \
                                        (x < 7 and self.walls[x+1,y+1] == 1)) or \
                            (y < 6 and self.walls[x,y+2] == 2)
                if touch_up or touch_mid or touch_down:
                    vert[n] = 1
        
        #Touching Pawns
        for pos in [self.p1pos,self.p2pos]:
            x,y = pos
            n = 8*x+y
            #top left
            if x > 0 and y > 0:
                horz[n-8-1] = 1
                vert[n-8-1] = 1
            #bottom left
            if x > 0 and y < 8:
                horz[n-8] = 1
                vert[n-8] = 1
            #top right
            if y > 0 and x < 8:
                horz[n-1] = 1
                vert[n-1] = 1
            #bottom right
            if x < 8 and y < 8:
                horz[n] = 1
                vert[n] = 1
        
        return horz+vert
    
    def probableValidWalls(self, player):
        valid = self.validWalls(player)
        probable = self.probableWalls()
        return [valid[i] and probable[i] for i in range(64*2)]
    
    def testNotClosing(self,x,y,orientation):
        """
        position: (x,y)
        orientation: 1 for horizontal, 2 for vertical
        
        Returns True if placing such a wall does not connect with a barrier on the right or does not connect
        with a barrier on the right. 
        
        This is checked as fast way to say a wall won't disconnect the board.
        """
        start = time.time()
        if orientation == 1:
            touch_left = (x == 0) or \
                        (x > 0 and (self.walls[x-1,y] == 2) or \
                                    (y > 0 and self.walls[x-1,y-1] == 2) or \
                                    (y < 7 and self.walls[x-1,y+1] == 2)) or \
                        (x > 1 and self.walls[x-2,y] == 1)
            ck1 = time.time()
            touch_mid = (y > 0 and (self.walls[x,y-1] == 2)) or \
                        (y < 7 and (self.walls[x,y+1] == 2))
            ck2 = time.time()
            touch_right = (x == 7) or \
                        (x < 7 and (self.walls[x+1,y] == 2) or \
                                    (y > 0 and self.walls[x+1,y-1] == 2) or \
                                    (y < 7 and self.walls[x+1,y+1] == 2)) or \
                        (x < 6 and self.walls[x+2,y] == 1)
            ck3 = time.time()
            warning = (touch_left and (touch_mid or touch_right)) or (touch_mid and touch_right)
            ck4 = time.time()
            return not warning
        
        if orientation == 2:
            touch_up = (y == 0) or \
                        (y > 0 and (self.walls[x,y-1] == 1) or \
                                    (x > 0 and self.walls[x-1,y-1] == 1) or \
                                    (x < 7 and self.walls[x+1,y-1] == 1)) or \
                        (y > 1 and self.walls[x,y-2] == 2)
            
            touch_mid = (x > 0 and (self.walls[x-1,y] == 1)) or \
                        (x < 7 and (self.walls[x+1,y] == 1))
             
            touch_down = (y == 7) or \
                        (y < 7 and (self.walls[x,y+1] == 1) or \
                                    (x > 0 and self.walls[x-1,y+1] == 1) or \
                                    (x < 7 and self.walls[x+1,y+1] == 1)) or \
                        (y < 6 and self.walls[x,y+2] == 2)
            warning = (touch_up and (touch_mid or touch_down)) or (touch_mid and touch_down)
            return not warning
            
    
    def testWallPlacement(self,x,y,orientation):
        """
        position: (x,y)
        orientation: 1 for horizontal, 2 for vertical
        
        Returns True if placing such a wall does not disconnect either player from their goal, else False.
        
        This is done by  first checking if the wall actuaully closes a gap then by
        a DFS of the boardGraph after the wall has been placed.
        """
        if self.testNotClosing(x,y,orientation):
            return True

        #create copy of the boardGraph after the wall placement
        graph = copy.deepcopy(self.boardGraph)
        
        if orientation == 1:
            graph[(x,y)].remove((x,y+1))
            graph[(x,y+1)].remove((x,y))
            graph[(x+1,y)].remove((x+1,y+1))
            graph[(x+1,y+1)].remove((x+1,y))
            
        else:
            graph[(x,y)].remove((x+1,y))
            graph[(x+1,y)].remove((x,y))
            graph[(x,y+1)].remove((x+1,y+1))
            graph[(x+1,y+1)].remove((x,y+1))
        
        # check if p1 is disconnected from goal
        p1connected = False
        visited = set()
        queue = [self.p1pos]
        while queue and not p1connected:
            vert = queue.pop(0)
            if vert[1] == 8:
                p1connected = True
            for i in graph[vert]:
                if i not in visited:
                    queue.append(i)
                    visited.add(i)
        
        # check if p2 is disconnected from goal
        p2connected = False
        visited = set()
        queue = [self.p2pos]
        while queue and not p2connected:
            vert = queue.pop(0)
            if vert[1] == 0:
                p2connected = True
            for i in graph[vert]:
                if i not in visited:
                    queue.append(i)
                    visited.add(i)
                    
        return p1connected and p2connected
    
    def getShortestPath(self, player):
        graph = copy.deepcopy(self.boardGraph)
        if player == 1:
            start = self.p1pos
            for x in range(9):
                graph[(x,8)].append('goal')
        else:
            start = self.p2pos
            for x in range(9):
                graph[(x,0)].append('goal')
        goal = 'goal'
        explored = []
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node not in explored:
                neighbours = graph[node]
                for neighbour in neighbours:
                    new_path = list(path)
                    if neighbour == goal:
                        return(new_path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                explored.append(node)
        return False
    
    def pawnsTouching(self):
        x1, y1 = self.p1pos
        x2, y2 = self.p2pos
        return (x1 == x2 and (y1 == y2-1 or y1 == y2+1)) or (y1 == y2 and (x1 == x2-1 or x1 == x2+1))
    
    def bestMove(self, player):
        if not self.pawnsTouching():
            path = self.getShortestPath(player)
            if not path:
                self.displayBoard()
                print("No path!")
            x,y = path[1]
            return 9*x+y
        else:
            best_move = (-1,-1)
            best_dist = 1000
            for x,y in self.validPawnMoves(player):
                tempboard = Board(self.vec)
                tempboard.movePawn(x,y, player)
                temp = tempboard.getShortestPath(player)
                if len(temp) < best_dist:
                    best_dist = len(temp)
                    best_move = temp[0]
                x,y = best_move
            return 9*x+y
    
    def placeWall(self,x,y,orientation):
        """
        x,y: location in 8x8 grid to place a wall. Note that this also corresponds to the lower right
        corner of a square on the game board.
        orientation: 1 for horz and 2 for vert
        
        updates self.walls and self.boardGraph
        """
        self.walls[x,y] = orientation
        
        if orientation == 1:
            self.boardGraph[(x,y)].remove((x,y+1))
            self.boardGraph[(x,y+1)].remove((x,y))
            self.boardGraph[(x+1,y)].remove((x+1,y+1))
            self.boardGraph[(x+1,y+1)].remove((x+1,y))
            
        else:
            self.boardGraph[(x,y)].remove((x+1,y))
            self.boardGraph[(x+1,y)].remove((x,y))
            self.boardGraph[(x,y+1)].remove((x+1,y+1))
            self.boardGraph[(x+1,y+1)].remove((x,y+1))
    
    def validActions(self, player):
        """
        returns a list of length 81+64+64 corresponding to pawn moves and wall placements (horz,vert).
        """
        moves = 81*[0]
        for x,y in self.validPawnMoves(player):
            moves[9*x+y] = 1
        return moves + self.validWalls(player)
    
    def probableValidActions(self,player):
        moves = 81*[0]
        for x,y in self.validPawnMoves(player):
            moves[9*x+y] = 1
        return moves + self.probableValidWalls(player)
    
    def takeAction(self, player, action):
        """
        Input:
            player: 1 or -1
            action: an int from 0 to 81+64+64 correpsonding to moving a pawn, placing a horz wall
            or placing a vert wall.

        Calls movePawn() which updates self.p1pos or placeWall() which updates self.walls and self.p1walls.
        """
        n = action
        if n <= 80:
            x,y = n // 9, n % 9
            self.movePawn(x,y,player)
        elif n <= 144:
            x,y = (n-81) // 8, (n-81) % 8
            self.placeWall(x,y,1)
            if player == 1:
                self.p1walls -= 1
            else:
                self.p2walls -= 1
        else:
            x,y = (n-145) // 8, (n-145) % 8
            self.placeWall(x,y,2)
            if player == 1:
                self.p1walls -= 1
            else:
                self.p2walls -= 1
    
    def getWinner(self, currentPlayer):
        """
        returns 1 if player 1 has won, -1 if player 2 has won, and 0 if neither has won.
        """
#         if self.p1walls == 0 and self.p2walls == 0:
#             # play the game out taking shortest moves???
#             a = len(self.getShortestPath(1))
#             b = len(self.getShortestPath(-1))
#             if b - a + 0.5*currentPlayer > 0:
#                 return 1
#             else:
#                 return -1
        if self.p1pos[1] == 8:
            return 1
        elif self.p2pos[1] == 0:
            return -1
        else:
            return 0
        
    def heuristicWinner(self, currentPlayer):
        """
        input:
            current player: 1 or -1
            
        Returns the player that will win if both players march towards the other side.
        """
        a = len(self.getShortestPath(1))
        b = len(self.getShortestPath(-1))
        if b - a + 0.5*currentPlayer > 0:
            return 1
        else:
            return -1
        
    def displayBoard(self):
        switch = True
        row = 0
        # iterate through rows
        while row < 9:
            square = True
            out = ''
            col = 0
            # iterate through a row with squares and vert walls
            if switch:
                out += ' '
                while col < 9:
                    if square:
                        if self.p1pos == (col,row):
                            out += ' 1 '
                        elif self.p2pos == (col,row):
                            out += ' 2 '
                        else:
                            out += '   '
                        col += 1  
                    else:
                        if row != 8 and self.walls[col - 1,row] == 2:
                            out += ' I '
                        elif row != 0 and self.walls[col - 1,row - 1] == 2:
                            out += ' I '
                        else:
                            out += ' : '  
                    square = not square
                row += 1
            # iterate through a row of horz walls
            else:
                while col < 9:
                    if col !=  8 and self.walls[col,row - 1] == 1:
                        out += '====='
                    elif col != 0 and self.walls[col - 1,row - 1] == 1:
                        out += '====='
                    else:
                        out += '.....'
                    if col != 8:
                        out += '+'
                    col += 1
            # print row and next row is a different type of row
            print(out)
            switch = not switch