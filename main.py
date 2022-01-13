from board import *
from ISMCTS import *

def PlayGame(nbIte, Players):
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(len(Players), Players, debug = False)
     
    while (state.getMoves() != []):
        state.show()
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if Players[state.currentPlayer == "ISMCTS_PARA"]:
                m = ISMCTS_para(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        elif Players[state.currentPlayer == "ISMCTS"]:
                m = ISMCTS(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        print ("Best Move: " + str(m) + "\n")
        state.doMove(m)

    state.getVictorious(True)

if __name__ == "__main__":
	PlayGame([20000, 20000, 20000], ["ISMCTS_PARA", "ISMCTS_PARA", "ISMCTS_PARA"])
