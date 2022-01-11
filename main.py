from board import *
from ISMCTS import *

def PlayGame():
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(3, ["ISMCTS", "ISMCTS", "ISMCTS"], debug = False)
    nbIte = 20000
    
    while (state.getMoves() != []):
        state.show()
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if state.currentPlayer == 0:
                m = ISMCTS_para(rootstate = state, itermax = nbIte, verbose = False)
        else:
                m = ISMCTS_para(rootstate = state, itermax = nbIte, verbose = False)
        print ("Best Move: " + str(m) + "\n")
        state.doMove(m)

    state.getVictorious(True)

if __name__ == "__main__":
	PlayGame()
