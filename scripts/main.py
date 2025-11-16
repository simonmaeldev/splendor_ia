import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import List, Dict, Tuple, Any
from splendor.board import *
from splendor.ISMCTS import *
from splendor.csv_exporter import export_game_to_csv
import sqlite3
from copy import deepcopy

def getColNames(cursor: sqlite3.Cursor, table: str) -> List[str]:
    cursor.execute("SELECT * FROM " + table)
    return [member[0] for member in cursor.description]

def getPlayersID(conn: sqlite3.Connection, cursor: sqlite3.Cursor, nbIte: List[int], Players: List[str]) -> List[int]:
    playersNames = [player + str(nbIte[i]) for i, player in enumerate(Players)]
    sqlInsert = '''INSERT OR IGNORE INTO Player (Name) VALUES (?)'''
    sqlSelectID = '''SELECT IDPlayer FROM Player WHERE Name = ?'''
    playersID = []
    print(playersNames)
    for n in playersNames:
        cursor.execute(sqlInsert, (n,))
        for row in cursor.execute(sqlSelectID, (n,)):
            playersID.append(row[0])
    return playersID

def createGame(cursor: sqlite3.Cursor, playersID: List[int], state: Board, winner: int) -> int:
    sqlInsert = '''INSERT INTO Game (NbPlayers, P1, P2, P3, P4, VictoryPoints, NbTurns, Winner)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
    data = [len(playersID)] + [playersID[i] if i < len(playersID) else None for i in range(4)] + [state.players[winner].getVictoryPoints(), state.nbTurn, winner]
    cursor.execute(sqlInsert, tuple(data))
    return cursor.lastrowid

def loadCards(cursor: sqlite3.Cursor) -> Dict[Tuple[int, int, int, int, int], int]:
    cards: Dict[Tuple[int, int, int, int, int], int] = {}
    sql = '''SELECT IDCard, CostWhite, CostBlue, CostGreen, CostRed, CostBlack FROM Card'''
    for row in cursor.execute(sql):
        cards[(row[1], row[2], row[3], row[4], row[5])] = row[0]
    return cards

def loadCharacters(cursor: sqlite3.Cursor) -> Dict[Tuple[int, int, int, int, int], int]:
    characters: Dict[Tuple[int, int, int, int, int], int] = {}
    sql = '''SELECT IDCharacter, CostWhite, CostBlue, CostGreen, CostRed, CostBlack FROM Character'''
    for row in cursor.execute(sql):
        characters[(row[1], row[2], row[3], row[4], row[5])] = row[0]
    return characters

def saveGamesState(cursor: sqlite3.Cursor, history: List[Any], cards: Dict[Tuple[int, int, int, int, int], int],
                  characters: Dict[Tuple[int, int, int, int, int], int], gameID: int,
                  deck_remaining_history: List[List[int]]) -> None:
    """
    Save game state with deck remaining counts.

    Args:
        history: List of game states [turn, currentPlayer, tokens, displayedCards1-3, characters]
        deck_remaining_history: List of [deck1_remaining, deck2_remaining, deck3_remaining] for each turn
    """
    colnames = getColNames(cursor, "StateGame")[1:]
    sqlInsert = '''INSERT INTO StateGame (''' + ", ".join(colnames) + ''')
    VALUES(''' + ", ".join(['?'] * len(colnames)) + ''')'''
    for i, s in enumerate(history):
        # get cards ID and complete by none if there isn't 4 cards for this level
        cardsID = [cards[tuple(c.cost)] for c in s[3]] + [None] * (4-len(s[3])) + [cards[tuple(c.cost)] for c in s[4]] + [None] * (4-len(s[4])) + [cards[tuple(c.cost)] for c in s[5]] + [None] * (4-len(s[5]))
        charactersID = [characters[tuple(c.cost)] for c in s[6]] + [None] * (5-len(s[6]))
        # Add deck remaining counts
        deck_remaining = deck_remaining_history[i] if i < len(deck_remaining_history) else [0, 0, 0]
        data = [gameID, s[0], s[1]] + s[2] + cardsID + charactersID + deck_remaining
        cursor.execute(sqlInsert, tuple(data))

def savePlayerState(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[List[int]],
                   board_states: List[Any], historyActionPlayers: List[List[Move]],
                   cards: Dict[Tuple[int, int, int, int, int], int]) -> None:
    """
    Save player state with enriched data (victory points, reductions, reserved cards).
    Reads state directly from Player objects instead of reconstructing from history.
    """
    colnames = getColNames(cursor, "StatePlayer")[1:]
    sqlInsert = '''INSERT INTO StatePlayer (''' + ", ".join(colnames) + ''')
    VALUES(''' + ", ".join(['?'] * len(colnames)) + ''')'''

    # Read state directly from board_states
    for turn, tokens in enumerate(history):
        player = board_states[turn].players[playerPos]
        vp = player.vp
        reductions = player.reductions.copy()

        # Get reserved card IDs
        reserved_cards = player.reserved
        reserved_ids = []
        for card in reserved_cards[:3]:
            try:
                card_id = cards[tuple(card.cost)]
                reserved_ids.append(card_id)
            except KeyError:
                # Card not in lookup, skip it
                pass
        reserved_ids_padded = reserved_ids + [None] * (3 - len(reserved_ids))

        # Build data row: gameID, turn, playerPos, tokens(6), vp, reductions(5), reserved(3)
        data = [gameID, turn, playerPos] + list(tokens) + [vp] + reductions + reserved_ids_padded
        cursor.execute(sqlInsert, tuple(data))

def savePlayerActions(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[Move],
                      cards: Dict[Tuple[int, int, int, int, int], int],
                      characters: Dict[Tuple[int, int, int, int, int], int],
                      board_states: List[Board]) -> None:
    colnames = getColNames(cursor, "Action")[1:]
    sqlInsert = "INSERT INTO Action (" + ", ".join(colnames) + ") VALUES (" + ", ".join(['?']*len(colnames)) + ")"
    for turn, a in enumerate(history):
        if a.actionType in [BUILD, RESERVE]:
            take = [None] * 6 if a.actionType == BUILD else TAKEONEGOLD
            give = [None] * 6
            # Get actual card object for both visible and top-deck reserves
            if a.actionType == RESERVE and isinstance(a.action, int):
                # Top-deck reserve: a.action is deck level (0, 1, or 2)
                # The card is drawn from the top of the deck and becomes visible only to this player
                # Get card from the deck BEFORE it was removed (board_states[turn] = state before action)
                board = board_states[turn]
                if a.action < len(board.decks) and len(board.decks[a.action]) > 0:
                    card = board.decks[a.action][0]  # Top card of the specified deck level
                    cardID = cards[tuple(card.cost)]
                else:
                    # Deck empty - shouldn't happen, but handle gracefully
                    cardID = None
            else:
                # Visible card reserve or build: a.action is already a Card object
                cardID = cards[tuple(a.action.cost)]
            characterID = characters[tuple(a.character.cost)] if a.character else None
        else:
            take = a.action
            give = a.tokensToRemove
            cardID = None
            characterID = None
        data = [gameID, turn, playerPos, a.actionType, cardID] + take + give + [characterID]
        cursor.execute(sqlInsert, tuple(data))

def saveIntoBdd(state: Board, winner: int, historyState: List[Any], historyPlayers: List[List[List[int]]],
               historyActionPlayers: List[List[Move]], nbIte: List[int], Players: List[str],
               deck_remaining_history: List[List[int]], board_states: List[Board]) -> None:
    """
    Save game data to database with enriched state information.

    Args:
        deck_remaining_history: List of [deck1_remaining, deck2_remaining, deck3_remaining] for each turn
        board_states: List of Board objects (deep copies) for each turn
    """
    #connect to bdd
    conn = sqlite3.connect('data/games.db')
    cursor = conn.cursor()
    # load into memory
    cards = loadCards(cursor)
    characters = loadCharacters(cursor)
    # get playersID
    playersID = getPlayersID(conn, cursor, nbIte, Players)
    gameID = createGame(cursor, playersID, state, winner)
    # insert states of games and players
    saveGamesState(cursor, historyState, cards, characters, gameID, deck_remaining_history)
    for i, h in enumerate(historyPlayers):
        savePlayerState(cursor, gameID, i, h, board_states, historyActionPlayers, cards)
    # insert actions of players
    for i, ha in enumerate(historyActionPlayers):
        savePlayerActions(cursor, gameID, i, ha, cards, characters, board_states)
    # only one commit at the very end, because there's only one connection at the same time for my db, therefor no concurrent access so I don't care
    conn.commit()
    conn.close()
    print("saved game successfully")
    return gameID
    
def PlayGame(nbIte: List[int], Players: List[str]) -> None:
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(len(Players), Players, debug = False)
    historyPlayers: List[List[List[int]]] = []
    historyState: List[Any] = []
    historyActionPlayers: List[List[Move]] = [[] for _ in range(len(Players))]
    deck_remaining_history: List[List[int]] = []
    # For CSV export: store (board_state, move, turn_num) tuples
    states_and_actions: List[Tuple[Board, Move, int]] = []

    for p in range(len(Players)):
        # initial state, turn 0
        historyPlayers += [[state.getPlayerState(p)]]

    while (state.getMoves() != []):
        state.show()
        historyState.append(state.getState())

        # Capture deck remaining counts
        deck_remaining_history.append([len(state.deckLVL1), len(state.deckLVL2), len(state.deckLVL3)])

        currentPlayer = state.currentPlayer
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if Players[currentPlayer] == "ISMCTS_PARA":
                m = ISMCTS_para(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        elif Players[currentPlayer] == "ISMCTS":
                m = ISMCTS(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        print ("Best Move: " + str(m) + "\n")

        # CRITICAL: Store deep copy of state BEFORE executing action (for CSV export)
        state_copy = deepcopy(state)
        states_and_actions.append((state_copy, m, state.nbTurn))

        # CRITICAL: Capture state BEFORE executing action (for ML training)
        historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))
        historyActionPlayers[currentPlayer].append(m)
        # Execute action
        state.doMove(m)

    state.show()
    winner = state.getVictorious(True)

    # Extract board states from states_and_actions for database save
    board_states = [state_copy for state_copy, move, turn_num in states_and_actions]

    # Save to database
    gameID = saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers,
                        nbIte, Players, deck_remaining_history, board_states)

    # Export to CSV AFTER database save (to use real game ID)
    try:
        from pathlib import Path
        # Use absolute path for CSV export to ensure correct location
        project_root = Path(__file__).parent.parent
        csv_output_dir = project_root / 'data' / 'games'
        export_game_to_csv(gameID, len(Players), states_and_actions, str(csv_output_dir))
        print(f"Exported game {gameID} to CSV successfully")
    except Exception as e:
        print(f"Warning: Failed to export game to CSV: {e}")

if __name__ == "__main__":
    # Use the new safe data collection system
    try:
        from data_collector import run_data_collection
        run_data_collection(
            config_path='data/simulation_config.txt',
            db_path='data/games.db',
            log_path='data/simulation_log.txt'
        )
    except ImportError:
        print("Error: Could not import data_collector module")
        print("Falling back to simple mode...")
        # Fallback to simple execution
        nbParties = 100
        while(True):
            try:
                PlayGame([1000, 1000, 1000], ["ISMCTS_PARA", "ISMCTS_PARA", "ISMCTS_PARA"])
                nbParties = nbParties - 1
            except(Exception):
                pass
