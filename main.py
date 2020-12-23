from custom_operators import *
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

    def play(self, player):
        # take 3 differents tokens
        # take 2 tokens of the same color IF there's at least 4 tokens of this color in the bank
        # build a card
        # reserve a card and take 1 gold
        # check end turn

    def take3tokens(self, player):
        listToken = player.ask3tokens(self)
        if (sum(listToken) > 3):
            print("that's more than 3 tokens sir! put them down")
            return false
        result = substract(self.tokens, listToken)
        if any r < 0 for r in result:
            print("there's not enough token for that")
            return false
        else:
            return self.takeTokens(player, listToken)

    def take2tokens(self, player):
        listToken = player.ask2tokens(self)
        if (sum(listToken) != 2):
            print("that's not 2 tokens sir! Put them down")
            return false
        colorWanted = listToken.index(2)
        if self.tokens[colorWanted] < 4:
            print("there's no 4 tokens of more from the color " + strColor(colorWanted) + " you can't take 2 tokens from this color")
            return false
        else :
            return self.takeTokens(player,listToken)

    def takeTokens(self, player, listToken):
        # add tokens to player reserve
        player.tokens = add(player.token, listToken)
        # remove thoses tokens from the board
        self.token = substract(self.token, listToken)
        return true
    
    def build(self, player):
        card = player.askBuild(self)
        if player.canBuild(card):
            rgc = player.realGoldCost(card)
            player.tokens = substract(player.tokens, rgc)
            self.token = add(self.tokens, rgc)
            return true
        else :
            return false
        
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
        
    def ask3tokens(self, board):
        return null

    def ask2tokens(self, board):
        return null

    def askBuild(self, board):
        return null
    
    def getTotalBonus(self):
        bonus = [0] * 5
        for card in self.built:
            bonus[card.bonus] += 1
        return bonus
         
    def realCost(self, card):
        cardCostWithBonus = substract(card.cost, self.getTotalBonus())
        return [c if c > 0 else 0 for c in cardCostWithBonus]

    def convertToGold(self, cost):
        maxPlayer = [min(p,c) for p,c in zip(self.tokens, cost)]
        goldNeeded = sum(cost) - sum(maxPlayer)
        maxPlayer[GOLD] = goldNeeded
        return maxPlayer

    def realGoldCost(self,card):
        rc = self.realCost(card)
        return self.convertToGold(rc)
    
    def canBuild(self, card):
        cost = self.realGoldCost(card)
        return self.token[GOLD] >= cost[GOLD]
        
   
class Character:
    def __init__(self, victoryPoints, cost):
        self.vp = victoryPoints
        self.cost = cost

    def __str__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])
