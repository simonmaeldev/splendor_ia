from typing import List, Dict, Tuple, Any
from board import *
from ISMCTS import *
import sqlite3

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

def saveGamesState(cursor: sqlite3.Cursor, history: List[Any], cards: Dict[Tuple[int, int, int, int, int], int], characters: Dict[Tuple[int, int, int, int, int], int], gameID: int) -> None:
    colnames = getColNames(cursor, "StateGame")[1:]
    sqlInsert = '''INSERT INTO StateGame (''' + ", ".join(colnames) + ''')
    VALUES(''' + ", ".join(['?'] * len(colnames)) + ''')'''
    for s in history:
        # get cards ID and complete by none if there isn't 4 cards for this level
        cardsID = [cards[tuple(c.cost)] for c in s[3]] + [None] * (4-len(s[3])) + [cards[tuple(c.cost)] for c in s[4]] + [None] * (4-len(s[4])) + [cards[tuple(c.cost)] for c in s[5]] + [None] * (4-len(s[5]))
        charactersID = [characters[tuple(c.cost)] for c in s[6]] + [None] * (5-len(s[6]))
        data = [gameID, s[0], s[1]] + s[2] + cardsID + charactersID
        cursor.execute(sqlInsert, tuple(data))

def savePlayerState(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[List[int]]) -> None:
    colnames = getColNames(cursor, "StatePlayer")[1:]
    sqlInsert = '''INSERT INTO StatePlayer (''' + ", ".join(colnames) + ''')
    VALUES(''' + ", ".join(['?'] * len(colnames)) + ''')'''
    for turn, s in enumerate(history):
        data = [gameID, turn, playerPos] + s
        cursor.execute(sqlInsert, tuple(data))

def savePlayerActions(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[Move], cards: Dict[Tuple[int, int, int, int, int], int], characters: Dict[Tuple[int, int, int, int, int], int]) -> None:
    colnames = getColNames(cursor, "Action")[1:]
    sqlInsert = "INSERT INTO Action (" + ", ".join(colnames) + ") VALUES (" + ", ".join(['?']*len(colnames)) + ")"
    for turn, a in enumerate(history):
        if a.actionType in [BUILD, RESERVE]:
            take = [None] * 6 if a.actionType == BUILD else TAKEONEGOLD
            give = [None] * 6
            cardID = cards[tuple(a.action.cost)] #will fail if someone reserve a top deck
            characterID = characters[tuple(a.character.cost)] if a.character else None
        else:
            take = a.action
            give = a.tokensToRemove
            cardID = None
            characterID = None
        data = [gameID, turn, playerPos, a.actionType, cardID] + take + give + [characterID]
        cursor.execute(sqlInsert, tuple(data))

def saveIntoBdd(state: Board, winner: int, historyState: List[Any], historyPlayers: List[List[List[int]]], historyActionPlayers: List[List[Move]], nbIte: List[int], Players: List[str]) -> None:
    #connect to bdd
    conn = sqlite3.connect('games.db')
    cursor = conn.cursor()
    # load into memory
    cards = loadCards(cursor)
    characters = loadCharacters(cursor)
    # get playersID
    playersID = getPlayersID(conn, cursor, nbIte, Players)
    gameID = createGame(cursor, playersID, state, winner)
    # insert states of games and players
    saveGamesState(cursor, historyState, cards, characters, gameID)
    for i, h in enumerate(historyPlayers):
        savePlayerState(cursor, gameID, i, h)
    # insert actions of players
    for i, ha in enumerate(historyActionPlayers):
        savePlayerActions(cursor, gameID, i, ha, cards, characters)
    # only one commit at the very end, because there's only one connection at the same time for my db, therefor no concurrent access so I don't care
    conn.commit()
    conn.close()
    print("saved game successfully")
    
def PlayGame(nbIte: List[int], Players: List[str]) -> None:
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(len(Players), Players, debug = False)
    historyPlayers: List[List[List[int]]] = []
    historyState: List[Any] = []
    historyActionPlayers: List[List[Move]] = [[]] * len(Players)
    for p in range(len(Players)):
        # initial state, turn 0
        historyPlayers += [[state.getPlayerState(p)]]

    while (state.getMoves() != []):
        state.show()
        historyState.append(state.getState())
        currentPlayer = state.currentPlayer
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if Players[currentPlayer] == "ISMCTS_PARA":
                m = ISMCTS_para(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        elif Players[currentPlayer] == "ISMCTS":
                m = ISMCTS(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        print ("Best Move: " + str(m) + "\n")
        state.doMove(m)
        historyActionPlayers[currentPlayer].append(m)
        # get state at the end of the turn
        historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))

    state.show()
    winner = state.getVictorious(True)
    saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, nbIte, Players)

if __name__ == "__main__":
    nbParties = 100
    while(True):
        try:
            PlayGame([1000, 1000, 1000], ["ISMCTS_PARA", "ISMCTS_PARA", "ISMCTS_PARA"])
            nbParties = nbParties - 1
        except(Exception):
            pass
