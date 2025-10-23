import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from typing import Tuple
from splendor.constants import *
import sqlite3

conn: sqlite3.Connection = sqlite3.connect('data/games.db')
cur: sqlite3.Cursor = conn.cursor()

def insertCharacter(character: Tuple[int, int, int, int, int, int]) -> None:
    sqlCharacter = '''INSERT INTO Character (VictoryPoints, CostWhite, CostBlue, CostGreen, CostRed, CostBlack)
    VALUES (?, ?, ?, ?, ?, ?)'''
    cur.execute(sqlCharacter, character)

def insertAllCharacters() -> None:
    for c in CHARACTERS:
        character = tuple([c[0]] + c[1])
        insertCharacter(character)
    print("inserted all characters")

def insertCard(card: Tuple[str, int, int, int, int, int, int, int]) -> None:
    sqlCard = '''INSERT INTO Card (Bonus, CostWhite, CostBlue, CostGreen, CostRed, CostBlack, VictoryPoints, Level)
Values (?, ?, ?, ?, ?, ?, ?, ?)'''
    cur.execute(sqlCard, card)

def insertAllCards() -> None:
    allCards = DECK1 + DECK2 + DECK3
    for c in allCards:
        card = tuple([fullStrColor(c[1])] + c[2] + [c[0], c[3]])
        insertCard(card)
    print("inserted all cards")

insertAllCharacters()
insertAllCards()

conn.commit()
conn.close()

print("database initialization was successfull")
