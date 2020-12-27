# https://gist.github.com/kjlubick/8ea239ede6a026a61f4d
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
from node import *
import random

def ISMCTS(rootstate, itermax, verbose = False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
    """

    rootnode = Node()

    for i in range(itermax):
        if i%100 == 0: print(f"{i}/{itermax}")
        node = rootnode

        # Determinize
        state = rootstate.cloneAndRandomize(rootstate.getCurrentPlayer())

        # Select
        while state.getMoves() != [] and node.getUntriedMoves(state.getMoves()) == []: # node is fully expanded and non-terminal
            node = node.UCBSelectChild(state.getMoves())
            state.doMove(node.move)

        # Expand
        untriedMoves = node.getUntriedMoves(state.getMoves())
        if untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(untriedMoves) 
            player = state.getCurrentPlayer()
            state.doMove(m)
            node = node.addChild(m, player) # add child and descend tree

        # Simulate
        # print("simulate")
        while state.getMoves() != []: # while state is non-terminal
            state.doMove(random.choice(state.getMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.update(state)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print (rootnode.treeToString(0))
    else: print (rootnode.childrenToString())

    return max(rootnode.childNodes, key = lambda c: c.visits).move # return the move that was most visited
