from board import *
from ISMCTS import *
import sqlite3

def getColNames(cursor, table):
    cursor.execute("SELECT * FROM " + table)
    return [member[0] for member in cursor.description]

def getPlayersID(conn, cursor, nbIte, Players):
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

def createGame(cursor, playersID, state, winner):
    sqlInsert = '''INSERT INTO Game (NbPlayers, P1, P2, P3, P4, VictoryPoints, NbTurns, Winner)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)'''
    data = [len(playersID)] + [playersID[i] if i<len(playersID) else None for i in range(4)] + [state.players[winner].getVictoryPoints(), state.nbTurn, winner]
    cursor.execute(sqlInsert, tuple(data))
    return cursor.lastrowid

def loadCards(cursor):
    cards = {}
    sql = '''SELECT IDCard, CostWhite, CostBlue, CostGreen, CostRed, CostBlack FROM Card'''
    for row in cursor.execute(sql):
        cards[(row[1], row[2], row[3], row[4], row[5])] = row[0]
    return cards

def loadCharacters(cursor):
    characters = {}
    sql = '''SELECT IDCharacter, CostWhite, CostBlue, CostGreen, CostRed, CostBlack FROM Character'''
    for row in cursor.execute(sql):
        characters[(row[1], row[2], row[3], row[4], row[5])] = row[0]
    return characters

def saveGamesState(cursor, history, cards, characters, gameID):
    colnames = getColNames(cursor, "StateGame")[1:]
    sqlInsert = '''INSERT INTO StateGame (''' + ", ".join(colnames) + ''') 
    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    for s in history:
        # get cards ID and complete by none if there isn't 4 cards for this level
        cardsID = [cards[tuple(c.cost)] for c in s[3]] + [None] * (4-len(s[3])) + [cards[tuple(c.cost)] for c in s[4]] + [None] * (4-len(s[4])) + [cards[tuple(c.cost)] for c in s[5]] + [None] * (4-len(s[5]))
        charactersID = [characters[tuple(c.cost)] for c in s[6]] + [None] * (5-len(s[6]))
        data = [gameID, s[0], s[1]] + s[2] + cardsID + charactersID
        cursor.execute(sqlInsert, tuple(data))

def savePlayerState(cursor, gameID, playerID, playerPos, history, cards, characters):
    colnames = getColNames(cursor, "StatePlayer")[1:]
    sqlInsert = '''INSERT INTO StatePlayer (''' + ", ".join(colnames) + ''') VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    for turn, s in enumerate(history):
        cardsID = [cards[tuple(c.cost)] for c in s[3]] + [None] * (3 - len(s[3]))
        charactersID = [characters[tuple(c.cost)] for c in s[4]] + [None] * (5 - len(s[4]))
        data = [gameID, turn, playerPos, playerID, s[0]] + s[1] + s[2] + cardsID + charactersID
        cursor.execute(sqlInsert, tuple(data))
       
def saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, nbIte, Players):
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
        savePlayerState(cursor, gameID, playersID[i], i, h, cards, characters)
    # insert actions of players

    # only one commit at the very end, because there's only one connection at the same time for my db, therefor no concurrent access so I don't care
    conn.commit()
    conn.close()
    print("saved game successfully")
    
def PlayGame(nbIte, Players):
    """ Play a sample game between two ISMCTS players.
    """
    state = Board(len(Players), Players, debug = False)
    historyPlayers = []
    historyState = []
    historyActionPlayers = [[]] * len(Players)
    for p in range(len(Players)):
        # initial state, turn 0
        historyPlayers += [[state.getPlayerState(p)]]
    
    while (state.getMoves() != []):
        state.show()
        historyState.append(state.getState())
        currentPlayer = state.currentPlayer
        # Use different numbers of iterations (simulations, tree nodes) for different players
        if Players[currentPlayer == "ISMCTS_PARA"]:
                m = ISMCTS_para(rootstate = state, itermax = nbIte[state.currentPlayer], verbose = False)
        elif Players[currentPlayer == "ISMCTS"]:
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
    while(True):
        PlayGame([20000, 20000, 20000], ["ISMCTS_PARA", "ISMCTS_PARA", "ISMCTS_PARA"])
