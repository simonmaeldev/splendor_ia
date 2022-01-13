from constants import *
import sqlite3

conn = sqlite3.connect('games.db')
cur = conn.cursor()

def insertCharacter(character):
    sqlCharacter = '''INSERT INTO Character (VictoryPoints, CostWhite, CostBlue, CostGreen, CostRed, CostBlack)
    VALUES (?, ?, ?, ?, ?, ?)'''
    cur.execute(sqlCharacter, character)
    conn.commit()
    
def insertAllCharacters():
    for c in CHARACTERS:
        character = tuple([c[0]] + c[1])
        insertCharacter(character)
    print("inserted all characters")

def insertCard(card):
    sqlCard = '''INSERT INTO Card (Bonus, CostWhite, CostBlue, CostGreen, CostRed, CostBlack, VictoryPoints, Level)
Values (?, ?, ?, ?, ?, ?, ?, ?)'''
    cur.execute(sqlCard, card)
    conn.commit()

def insertAllCards():
    allCards = DECK1 + DECK2 + DECK3
    for c in allCards:
        card = tuple([fullStrColor(c[1])] + c[2] + [c[0], c[3]])
        insertCard(card)
    print("inserted all cards")

insertAllCharacters()
insertAllCards()

conn.close()

print("database initialization was successfull")
