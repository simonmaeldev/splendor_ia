from constants import *
from random import shuffle

class Board:
    def __init__(self, nbPlayer):
        self.deckLVL1 = [Card(c[0], c[1], c[2], c[3]) for c in DECK1]
        shuffle(self.deckLVL1)
        self.deckLVL2 = [Card(c[0], c[1], c[2], c[3]) for c in DECK2]
        shuffle(self.deckLVL2)
        self.deckLVL3 = [Card(c[0], c[1], c[2], c[3]) for c in DECK3]
        shuffle(self.deckLVL3)
        nbToken = self.getNbMaxTokens(nbPlayer)
        # color order : white blue green red black gold
        self.tokens = [nbToken] * 5 + [5]
        self.characters = [Character(c[0], c[1]) for c in CHARACTERS]
        shuffle(self.characters)
        self.characters = self.characters[:nbPlayer + 1]
        self.lvl1 = [self.deckLVL1.pop(0) for i in range(0,4)]
        self.lvl2 = [self.deckLVL2.pop(0) for i in range(0,4)]
        self.lvl3 = [self.deckLVL3.pop(0) for i in range(0,4)]
        self.players = [Player(i) for i in range(1,nbPlayer + 1)]

    def getNbMaxTokens(self, nbPlayer):
       if nbPlayer == 2:
           return 4
       elif nbPlayer == 3:
           return 5
       else : return 7

class Card:
    def __init__(self, victoryPoints, bonus, cost, lvl):
        self.vp = victoryPoints
        self.bonus = bonus
        self.cost = cost
        self.lvl = lvl

    def __str__(self):
        return "vp: {}, color: {}, cost: {}, lvl{}".format(self.vp, strColor(self.bonus), [str(self.cost[i]) + strColor(i) for i in range(0,5)], self.lvl)

class Player:
    def __init__(self, name):
        self.name = name
        self.tokens = [0] * 6
        self.built = []
        self.reserved = []
        self.characters = []
        
class Character:
    def __init__(self, victoryPoints, cost):
        self.vp = victoryPoints
        self.cost = cost

    def __str__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])

board = Board(2)
print([str(c) for c in board.lvl1])
print([str(c) for c in board.characters])
print(board.tokens[WHITE])
