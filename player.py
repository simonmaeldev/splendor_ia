from constants import *
from custom_operators import *

class Player:
    def __init__(self, name):
        self.name = name
        self.tokens = [0] * 6
        self.built = []
        self.reserved = []
        self.characters = []

    def getShow(self):
        bonus = self.getTotalBonus()
        return f"{self.name} vp:{self.getVictoryPoints()} reserved:{len(self.reserved)} tokens:{bcolors.WHITE}{self.tokens[WHITE]} {bcolors.BLUE}{self.tokens[BLUE]} {bcolors.GREEN}{self.tokens[GREEN]} {bcolors.RED}{self.tokens[RED]} {bcolors.BLACK}{self.tokens[BLACK]} {bcolors.YELLOW}{self.tokens[GOLD]}{bcolors.RESET} bonus:[{bcolors.WHITE}{bonus[WHITE]} {bcolors.BLUE}{bonus[BLUE]} {bcolors.GREEN}{bonus[GREEN]} {bcolors.RED}{bonus[RED]} {bcolors.BLACK}{bonus[BLACK]}{bcolors.RESET}]"

    def show(self):
        bonus = self.getTotalBonus()
        print(f"{self.name} vp:{self.getVictoryPoints()} reserved:{len(self.reserved)} tokens:{bcolors.WHITE}{self.tokens[WHITE]} {bcolors.BLUE}{self.tokens[BLUE]} {bcolors.GREEN}{self.tokens[GREEN]} {bcolors.RED}{self.tokens[RED]} {bcolors.BLACK}{self.tokens[BLACK]} {bcolors.YELLOW}{self.tokens[GOLD]}{bcolors.RESET} bonus:[{bcolors.WHITE}{bonus[WHITE]} {bcolors.BLUE}{bonus[BLUE]} {bcolors.GREEN}{bonus[GREEN]} {bcolors.RED}{bonus[RED]} {bcolors.BLACK}{bonus[BLACK]}{bcolors.RESET}]")
        
    def showReserved(self):
        cards = "" if (len(self.reserved) > 0) else "none"
        for c in self.reserved:
            cards += "|" + c.getShow() + "|"
        return cards
    
    def askAction(self, board):
        return null
    
    def ask3tokens(self, board):
        return null

    def ask2tokens(self, board):
        return null

    def askBuild(self, board):
        return null
    
    def askReserve(self, board):
        return null
    
    def askTokenToRemove(self, board):
        return null
    
    def takeCharacter(self, board):
        return null
    
    def getVictoryPoints(self):
        vp = sum([card.vp for card in self.built]) + sum([c.vp for c in self.characters])
        return vp
        
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
        
    def gotToManyTokens(self):
        return sum(self.tokens) > MAX_NB_TOKENS
