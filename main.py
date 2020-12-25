from custom_operators import *
from constants import *
from random import shuffle
from player import *
from characters import *
from cards import *

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
        self.displayedCards = [[self.deckLVL1.pop(0) for i in range(0,4)], [self.deckLVL2.pop(0) for i in range(0,4)], [self.deckLVL3.pop(0) for i in range(0,4)]]
        self.players = [Player(str(i)) for i in range(1,nbPlayer + 1)]
        self.endGame = False
        self.currentPlayer = 0
        self.nbTurn = 1

    def show(self):
        print("tour n" + str(self.nbTurn))
        print()
        # tokens from board
        print(' '.join(self.getShowTokens()) + bcolors.RESET)
        # cards
        for lvl in range(0,3):
            cards = ""
            for card in self.displayedCards[lvl]:
                cards += "|" + card.getShow() + "|"
            print(f"lvl{lvl+1} : " + cards + f" ({len(self.decks[lvl])})")
        # characters
        chars = ""
        for c in self.characters:
            chars += "|" + c.getShow() + "|"
        print("nobles:"+chars)
        print()
        # players
        for i, player in enumerate(self.players):
            output = ""
            if i == self.currentPlayer:
                output += f"{bcolors.BOLD}-> "
            output += player.getShow() + bcolors.RESET
            print(output)
        # current player
        print("\nreserved cards of current player: " + self.getCurrentPlayer().showReserved())

    def getCurrentPlayer(self):
        return self.players[self.currentPlayer]
        
    def getShowTokens(self):
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.tokens)]
    
    def getNbMaxTokens(self, nbPlayer):
       if nbPlayer == 2:
           return 4
       elif nbPlayer == 3:
           return 5
       else : return 7

    def turn(self, player):
        print("==========================")
        self.show()
        res = False
        while not res:
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
                res = False

        # while the player got to many tokens, remove them
        while player.gotToManyTokens():
            self.takeOneTokenFromPlayer(player)
        player.takeCharacter(self)
        self.checkEndGame(player)
        
    def play(self):
        while True:
            self.turn(self.players[self.currentPlayer])
            self.currentPlayer += 1
            if self.currentPlayer >= len(self.players):
                if self.endGame:
                    break
                else:
                    self.currentPlayer = 0
                    self.nbTurn += 1
        self.printVictorious()

    def printVictorious(self):
        vp = [player.getVictoryPoints() for player in self.players]
        indices = [i for i, p in enumerate(vp) if p == max(vp)]
        if len(indices) == 1:
            winner = indices[0]
        else:
            print("some players have the same amout of victory points. the winner is the one with less cards")
            nbCard = [len(self.players[i].built) for i in indices]
            winner = indices[nbCard.index(min(nbCard))]
        print("player " + self.players[winner].name + " won")
        return winner

    def takeOneTokenFromPlayer(self, player):
        color = player.askTokenToRemove(self)
        if player.tokens[color] > 0:
            player.tokens[color] -= 1
        else:
            return False
        self.tokens[color] += 1

    def checkEndGame(self, player):
       if player.getVictoryPoints() >= VP_GOAL:
           print(f"{bcolors.RED + bcolors.BOLD}WARNING it's the last turn ! {bcolors.RESET}")
           self.endGame = True
   
    def take3tokens(self, player):
        listToken = player.ask3tokens(self)
        if (sum(listToken) > 3):
            print("that's more than 3 tokens sir! put them down")
            return False
        result = substract(self.tokens, listToken)
        if any (r < 0 for r in result):
            print("there's not enough token for that")
            return False
        else:
            return self.takeTokens(player, listToken)

    def take2tokens(self, player):
        listToken = player.ask2tokens(self)
        if listToken == None:
            return False
        if (sum(listToken) != 2):
            print("that's not 2 tokens sir! Put them down")
            return False
        colorWanted = listToken.index(2)
        if self.tokens[colorWanted] < 4:
            print("there's no 4 tokens of more from the color " + strColor(colorWanted) + " you can't take 2 tokens from this color")
            return False
        else :
            return self.takeTokens(player,listToken)

    # return True if there was enough tokens to take
    def takeTokens(self, player, listToken):
        enoughTokens = True
        for color in range(0,6):
            while listToken[color] > 0 and self.tokens[color] > 0:
                player.tokens[color] += 1
                self.tokens[color] -= 1
                listToken[color] -= 1
            enoughTokens = enoughTokens and listToken[color] == 0
        return enoughTokens
            

    def removeCard(self, card):
        visibles = flatten(self.displayedCards)
        if card in visibles:
            self.displayedCards[card.lvl - 1].remove(card)
            if len(self.decks[card.lvl - 1]) > 0:
                self.displayedCards[card.lvl - 1].append(self.decks[card.lvl - 1].pop(0))
        else:
            self.decks[card.lvl - 1].remove(card)
    
    def build(self, player):
        card = player.askBuild(self)
        if player.canBuild(card):
            rgc = player.realGoldCost(card)
            player.tokens = substract(player.tokens, rgc)
            self.tokens = add(self.tokens, rgc)
            if card in player.reserved:
                player.reserved.remove(card)
            else :
                self.removeCard(card)
            player.built.append(card)
            return True
        else :
            print("you can't build that at the moment, sorry")
            return False
        
    def reserve(self, player):
        if len(player.reserved) >= 3:
            return False
        else :
            card = player.askReserve(self)
            self.removeCard(card)
            player.reserved.append(card)
            self.takeTokens(player, TAKEONEGOLD.copy())
            return True
        
b = Board(2)
b.play()
