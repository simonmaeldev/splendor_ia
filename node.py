# https://gist.github.com/kjlubick/8ea239ede6a026a61f4d
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
from math import *

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
    """
    def __init__(self, move = None, parent = None, playerJustMoved = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.avails = 1
        self.playerJustMoved = playerJustMoved # the only part of the state that the Node needs later

    def getUntriedMoves(self, legalMoves):
        """ Return the elements of legalMoves for which this node does not have children.
        """        
        # Find all moves for which this node *does* have children
        triedMoves = [child.move for child in self.childNodes]
        
        # Return all moves that are legal but have not been tried yet
        return [move for move in legalMoves if move not in triedMoves]
        
    def UCBSelectChild(self, legalMoves, exploration = 0.7):
        """ Use the UCB1 formula to select a child node, filtered by the given list of legal moves.
        exploration is a constant balancing between exploitation and exploration, with default value 0.7 (approximately sqrt(2) / 2)
        """
                
        # Filter the list of children by the list of legal moves
        legalChildren = [child for child in self.childNodes if child.move in legalMoves]
                
        # Get the child with the highest UCB score
        s = max(legalChildren, key = lambda c: float(c.wins)/float(c.visits) + exploration * sqrt(log(c.avails)/float(c.visits)))
                
        # Update availability counts -- it is easier to do this now than during backpropagation
        for child in legalChildren:
            child.avails += 1

        # Return the child selected above
        return s
        
    def addChild(self, m, p):
        """ Add a new child node for the move m.
        Return the added child node
        """
        n = Node(move = m, parent = self, playerJustMoved = p)
        self.childNodes.append(n)
        return n
        
    def update(self, terminalState):
        """ Update this node - increment the visit count by one, and increase the win count by the result of terminalState for self.playerJustMoved.
        """
        self.visits += 1
        if self.playerJustMoved != None:
            # print(f"{self.playerJustMoved.name} {terminalState.getVictorious()}")
            self.wins += terminalState.getResult(self.playerJustMoved)

    def __repr__(self):
        return "[M:%s W/V/A: %4i/%4i/%4i]" % (self.move, self.wins, self.visits, self.avails)

    def treeToString(self, indent):
        """ Represent the tree as a string, for debugging purposes.
        """
        s = self.indentString(indent) + str(self)
        for c in self.childNodes:
            s += c.treeToString(indent+1)
        return s

    def indentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def childrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s

    def __eq__(self, other):
        return other != None and self.move == other.move

    def __ne__(self, other):
        return not self.__eq__(other)
