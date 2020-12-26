from constants import *
from custom_operators import *

class Player:
    def __init__(self, name, IA):
        self.name = name
        self.tokens = [0] * 6
        self.built = []
        self.reserved = []
        self.characters = []
        self.IA = IA
        # action : [type, card/tokenList, character/tokens to remove if too much]
        self.action = []

    def isHuman():
        return self.IA == None

    def getShowBonus(self):
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.getTotalBonus())]
    
    def getShowTokens(self):
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.tokens)]

    def getShow(self):
        bonus = self.getTotalBonus()
        tokens = self.getShowTokens()
        return f"{self.name} vp:{self.getVictoryPoints()} reserved:{len(self.reserved)} tokens:{' '.join(tokens)}{bcolors.RESET} bonus:[{' '.join(self.getShowBonus())}]"

    def show(self):
        print(self.getShow())
        
    def showReserved(self):
        cards = "" if (len(self.reserved) > 0) else "none"
        for c in self.reserved:
            cards += "|" + c.getShow() + "|"
        return cards
    
    def askAction(self, board):
        if self.isHuman():
            action = -1
            while action < 0 or action > 3:
                print("0 : build\n1 : reserve\n2 : take 2 tokens of the same color\n3 : take 3 differents tokens")
                action = int(input())
            return action
        else:
            self.action = IA.getAction(board)
            return action[0]

    def getFinalAction(self):
        return self.action[1]
    
    def getComplementaryAction(self):
        return self.action[2]
    
    def ask3tokens(self, board):
        if self.isHuman():
            valide = False
            while not valide:
                print("place an x under the colors you want, space if you don't want it")
                print(''.join(board.getShowTokens()[:5]) + bcolors.RESET)
                choice = input() + "      "
                choice = choice[:5]
                nbSelected = choice.count('x')
                nbPossible = 5 - board.tokens[:5].count(0)
                valide = (nbSelected <= min(nbPossible, 3))
            return [1 if c == 'x' else 0 for c in choice] + [0]
        else:
            return self.getFinalAction()

    def ask2tokens(self, board):
        if self.isHuman():
            valide = False
            if not any(t >= 4 for t in board.tokens[:5]):
                print("you cant take 2 tokens of the same color, choose an other action")
                return None
            while not valide:
                print("place an x under the color you want to take 2 tokens, space for the others")
                print(''.join(board.getShowTokens()[:5]) + bcolors.RESET)
                choice = input() + "      "
                choice = choice[:5]
                nbSelected = choice.count('x')
                valide = nbSelected == 1 and board.tokens[choice.index('x')] >= 4
            return [2 if c == 'x' else 0 for c in choice] + [0]
        else:
            return self.getFinalAction()

    def askBuild(self, board):
        if self.isHuman():
            valide = False
            while not valide:
                print("choose the lvl of the card (4 for your reserved cards)")
                lvl = int(input())
                print("which one? 1: most left one, 4: most right one")
                nb = int(input())
                allCards = board.displayedCards + [self.reserved]
                valide = lvl in range(1,5) and nb in range(1, len(allCards[lvl - 1]) + 1)
            return allCards[lvl - 1][nb - 1]
        else:
            return self.getFinalAction()
    
    def askReserve(self, board):
        if self.isHuman():
            valide = False
            while not valide:
                print("choose the lvl of the card")
                lvl = int(input())
                print("which one? 1: most left one, 4: most right one, 5: top deck")
                nb = int(input())
                valide = lvl in range(1,4) and (nb == 5 and board.decks[lvl - 1] or nb in range(1,len(board.displayedCards[lvl - 1]) + 1))
            return board.deck[lvl - 1][0] if nb == 5 else board.displayedCards[lvl - 1][nb - 1]
        else:
            return self.getFinalAction()
    
    def askTokenToRemove(self, board):
        if self.isHuman():
            valide = False
            while not valide:
                print("you have too many tokens. Please place an x under the color of your choice, we will take one token from this color")
                print(''.join(self.getShowTokens()) + bcolors.RESET)
                choice = input() + "      "
                choice = choice[:6]
                valide = choice.count('x') == 1 and self.tokens[choice.index('x')] > 0
            return choice.index('x')
        else:
            willRemove = self.getComplementaryAction()
            color = willRemove.index(list(filter(lambda x: x > 0, willRemove))[0])
            willRemove[color] -= 1
            return color
    
    def takeCharacter(self, board):
        possible = list(filter(lambda c: all(color >=0 for color in substract(self.getTotalBonus(), c.cost)), board.characters))
        if possible:
            c = None
            if len(possible) == 1:
                c = possible[0]
            else:
                if self.isHuman():
                    print("you can have multiple noble this turn. which one do you want?")
                    for i, character in enumerate(possible):
                        print(str(i) + ": "+ character.getShow())
                    valide = False
                    while not valide:
                        choice = int(input())
                        valide = choice in range(0, len(possible))
                    c = possible[choice]
                else:
                    c = self.getComplementaryAction()
            self.characters.append(c)
            board.characters.remove(c)
    
    def getVictoryPoints(self):
        return sum([card.vp for card in self.built]) + sum([c.vp for c in self.characters])
        
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
        maxPlayer += [goldNeeded]
        return maxPlayer

    def realGoldCost(self,card):
        rc = self.realCost(card)
        return self.convertToGold(rc)
    
    def canBuild(self, card):
        cost = self.realGoldCost(card)
        return self.tokens[GOLD] >= cost[GOLD]
        
    def gotToManyTokens(self):
        return sum(self.tokens) > MAX_NB_TOKENS
