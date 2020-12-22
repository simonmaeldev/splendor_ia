class Board:
    def __init__(self, nbPlayer):
        self.deckLVL1 = null
        self.deckLVL2 = null
        self.deckLVL3 = null
        nbToken = self.getNbMaxTokens(nbPlayer)
        # color order : white blue green red black gold
        self.tokens = [nbToken] * 5 + [5]
        self.characters = []
        self.lvl1 = []
        self.lvl2 = []
        self.lvl3 = []
        self.players = [Player(i) for i in range(1,4)]

class Card:
    def __init__(self, victoryPoints, bonus, cost, lvl):
        self.vp = victoryPoints
        self.bonus = bonus
        self.cost = cost
        self.lvl = lvl

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
