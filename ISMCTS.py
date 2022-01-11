# https://gist.github.com/kjlubick/8ea239ede6a026a61f4d
# Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
# 
# Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
# remains in any distributed code.
from node import *
import random
from timeit import default_timer as timer

def ISMCTS(rootstate, itermax, verbose = False, returnTree = False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
    """

    # n'est pas parallele + ne conserve pas l'arbre des decisions
    start = timer()
    rootnode = Node()

    for i in range(itermax):
        node = rootnode

        # Determinize
        state = rootstate.cloneAndRandomize(rootstate.getCurrentPlayer())

        # Select
        #print("select")
        while state.getMoves() != [] and node.getUntriedMoves(state.getMoves()) == []: # node is fully expanded and non-terminal
            node = node.UCBSelectChild(state.getMoves())
            state.doMove(node.move)

        # Expand
        #print("expand")
        untriedMoves = node.getUntriedMoves(state.getMoves())
        if untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(untriedMoves) 
            player = state.getCurrentPlayer()
            state.doMove(m)
            node = node.addChild(m, player) # add child and descend tree

        # Simulate
        #print("simulate")
        while state.getMoves() != []: # while state is non-terminal
            state.doMove(random.choice(state.getMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.update(state)
            node = node.parentNode

    # Output some information about the tree - can be omitted
    #if (verbose): print (rootnode.treeToString(0))
    #else: print (rootnode.childrenToString())

    end = timer()
    #print(f'ismcts time : {end - start}')
    return rootnode if returnTree else max(rootnode.childNodes, key = lambda c: c.visits).move # return the move that was most visited

from multiprocessing import Pool, cpu_count

def ISMCTS_para(rootstate, itermax, verbose = False):
    """ Conduct an ISMCTS search for itermax iterations starting from rootstate.
            Return the best move from the rootstate.
    """
    start = timer()
    
    np = cpu_count()
    part_tree = map((lambda i: (rootstate, itermax//np, False, True)), range(np))

    with Pool(processes=np) as pool:
        #compute multiples tree at the same time
        tree = pool.starmap(ISMCTS, part_tree)
        # recompose tree
        print(f'time simulation : {timer() - start}')
        rootnode = tree[0]
        for t in tree[1:]:
            mergeTrees(rootnode, t)

    end = timer()
    print(f'total time: {end - start}') 
    #print (rootnode.childrenToString())
    return max(rootnode.childNodes, key = lambda c: c.visits).move # return the move that was most visited

def mergeTrees(originTree, addTree):
    for ac in addTree.childNodes:
        if ac in originTree.childNodes:
            #merge the two identical nodes
            oc = next(filter((lambda n: n == ac), originTree.childNodes))
            oc.wins += ac.wins
            oc.visits += ac.visits
            oc.avails += ac.avails
            mergeTrees(oc, ac)
        else:
            # add the node to the origin tree
            originTree.childNodes.append(ac)
            ac.parentNode = originTree
