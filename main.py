from board import *
from ISMCTS import *

def PlayGame():
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(2, ["ISMCTS", "ISMCTS"])

    while (state.getMoves() != []):
        state.show()
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if state.currentPlayer == 0:
                m = ISMCTS(rootstate = state, itermax = 2000, verbose = False)
        else:
                m = ISMCTS(rootstate = state, itermax = 100, verbose = False)
        print ("Best Move: " + str(m) + "\n")
        state.doMove(m)

    state.getVictorious(True)

if __name__ == "__main__":
	PlayGame()
