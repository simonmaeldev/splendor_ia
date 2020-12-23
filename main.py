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
        self.decks = [self.deckLVL1, self.deckLVL2, self.deckLVL3]
        nbToken = self.getNbMaxTokens(nbPlayer)
        # color order : white blue green red black gold
        self.tokens = [nbToken] * 5 + [5]
        self.characters = [Character(c[0], c[1]) for c in CHARACTERS]
        shuffle(self.characters)
        self.characters = self.characters[:nbPlayer + 1]
        self.lvl1 = [self.deckLVL1.pop(0) for i in range(0,4)]
        self.lvl2 = [self.deckLVL2.pop(0) for i in range(0,4)]
        self.lvl3 = [self.deckLVL3.pop(0) for i in range(0,4)]
        self.displayedCards [[self.deckLVL1.pop(0) for i in range(0,4)], [self.deckLVL2.pop(0) for i in range(0,4)], [self.deckLVL3.pop(0) for i in range(0,4)]]
        self.players = [Player(i) for i in range(1,nbPlayer + 1)]
        self.endGame = false
        self.currentPlayer = 0
        self.nbTurn = 1

    def getNbMaxTokens(self, nbPlayer):
       if nbPlayer == 2:
           return 4
       elif nbPlayer == 3:
           return 5
       else : return 7

    def turn(self, player):
        res = true
        while !res:
            action = player.askAction(self)
            if action == BUILD:
                res = self.build(player)
            elif action == RESERVE:
                res = self.reserve(player)
            elif action == TAKE2:
                res = self.take2tokens(player)
            elif action == TAKE3:
                res = self.take3tokens(player)
            else:
                res = false

        # while the player got to many tokens, remove them
        while true:
            if player.gotToManyTokens():
                self.takeOneTokenFromPlayer(player)
            else:
                break
        player.takeCharacter(self)
        self.checkEndGame(player)
        
    def play(self):
        while true:
            self.turn(self.players[self.currentPlayer])
            self.currentPlayer += 1
            if self.currentPlayer >= len(self.players):
                if self.endGame:
                    break
                else:
                    self.currentPlayer = 0
                    self.turn += 1
        self.printVictorious()

    def printVictorious(self):
        vp = [player.getVitctoryPoints() for player in self.players]
        indices = [i for i, p in enumerate(my_list) if p == map(vp)]
        if len(indices) == 1:
            winner = indices[0]
        else:
            nbCard = [len(self.players[i].built) for i in indices]
            winner = indices[nbCard.index(min(nbCard))]
        print("player " + self.players[winner].name + " won")
        return winner

    def takeOneTokenFromPlayer(self, player):
        color = player.askTokenToRemove(self)
        player.tokens[color] -= 1
        self.tokens[color] += 1

    def checkEndGame(self, player):
       if player.getVictoryPoints >= VP_GOAL:
           self.endGame = true
   
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

    # return true if there was enough tokens to take
    def takeTokens(self, player, listToken):
        enoughTokens = true
        for color in range(0,6):
            while listToken[color] > 0 and self.tokens[color] > 0:
                player.tokens[color] += 1
                self.tokens[color] -= 1
                listToken[color] -= 1
            enoughTokens = enoughTokens and listToken[color] == 0
        return enoughTokens
            

    def removeCard(self, card):
        self.displayedCards[card.lvl - 1].remove(card)
        if len(self.decks[card.lvl - 1]) > 0:
            self.displayedCards[card.lvl - 1].append(self.decks[card.lvl - 1].pop(0))
    
    def build(self, player):
        card = player.askBuild(self)
        if player.canBuild(card):
            rgc = player.realGoldCost(card)
            player.tokens = substract(player.tokens, rgc)
            self.token = add(self.tokens, rgc)
            if card in player.reserved:
                player.reserved.remove(card)
            else :
                self.removeCard(card)
            player.built.append(card)
            return true
        else :
            return false
        
    def reserve(self, player):
        if len(player.reserved) >= 3:
            return false
        else :
            card = player.askReserve(self)
            self.removeCard(card)
            player.reserve.append(card)
            self.takeTokens(player, TAKEONEGOLD)
            return true
        
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
   
class Character:
    def __init__(self, victoryPoints, cost):
        self.vp = victoryPoints
        self.cost = cost

    def __str__(self):
        return "vp: {}, cost: {}".format(self.vp, [str(self.cost[i]) + strColor(i) for i in range(0,5)])
