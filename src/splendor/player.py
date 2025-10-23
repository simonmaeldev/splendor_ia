from typing import List, Optional, Any, TYPE_CHECKING
from .constants import *
from .custom_operators import *
import uuid

if TYPE_CHECKING:
    from .cards import Card
    from .characters import Character
    from .board import Board

class Player:
    def __init__(self, name: str, IA: Optional[str]) -> None:
        self.name: str = name
        self.tokens: List[int] = [0] * 6
        self.built: List['Card'] = []
        self.reserved: List['Card'] = []
        self.characters: List['Character'] = []
        self.IA: Optional[str] = IA
        # action : [type, card/tokenList, character/tokens to remove if too much]
        self.action: List[Any] = []
        self.idCustom: uuid.UUID = uuid.uuid1()

    def __eq__(self, other: object) -> bool:
        return other != None and self.idCustom == other.idCustom

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def isHuman(self) -> bool:
        return self.IA == None

    def haveUnseenCards(self) -> bool:
        return not all(c.isVisible() for c in self.reserved)

    def getUnseenCards(self) -> List['Card']:
        return list(filter(lambda card : not card.isVisible(), self.reserved))

    def removeUnseenCards(self) -> None:
        self.reserved = [card for card in self.reserved if card not in self.getUnseenCards()]
        
    def getShowBonus(self) -> List[str]:
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.getTotalBonus())]

    def getShowTokens(self) -> List[str]:
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.tokens)]

    def getShow(self) -> str:
        bonus = self.getTotalBonus()
        tokens = self.getShowTokens()
        return f"{self.name} vp:{self.getVictoryPoints()} reserved:{len(self.reserved)} tokens:{' '.join(tokens)}{bcolors.RESET} bonus:[{' '.join(self.getShowBonus())}]"

    def show(self) -> None:
        print(self.getShow())

    def showReserved(self) -> str:
        cards = "" if (len(self.reserved) > 0) else "none"
        for c in self.reserved:
            cards += "|" + c.getShow() + "|"
        return cards
    
    def askAction(self, board: 'Board') -> int:
        if self.isHuman():
            action = -1
            while action < 0 or action > 3:
                print("0 : build\n1 : reserve\n2 : take 2 tokens of the same color\n3 : take 3 differents tokens")
                action = int(input())
            return action
        else:
            self.action = IA.getAction(board)
            return action[0]

    def getFinalAction(self) -> Any:
        return self.action[1]

    def getComplementaryAction(self) -> Any:
        return self.action[2]

    def ask3tokens(self, board: 'Board') -> List[int]:
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

    def ask2tokens(self, board: 'Board') -> Optional[List[int]]:
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

    def askBuild(self, board: 'Board') -> 'Card':
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

    def askReserve(self, board: 'Board') -> 'Card':
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

    def askTokenToRemove(self, board: 'Board') -> int:
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
    
    def getVictoryPoints(self) -> int:
        return sum([card.vp for card in self.built]) + sum([c.vp for c in self.characters])

    def getTotalBonus(self) -> List[int]:
        bonus = [0] * 5
        for card in self.built:
            bonus[card.bonus] += 1
        return bonus

    def realCost(self, card: 'Card') -> List[int]:
        cardCostWithBonus = substract(card.cost, self.getTotalBonus())
        return [c if c > 0 else 0 for c in cardCostWithBonus]

    def convertToGold(self, cost: List[int]) -> List[int]:
        maxPlayer = [min(p,c) for p,c in zip(self.tokens, cost)]
        goldNeeded = sum(cost) - sum(maxPlayer)
        maxPlayer += [goldNeeded]
        return maxPlayer

    def realGoldCost(self, card: 'Card') -> List[int]:
        rc = self.realCost(card)
        return self.convertToGold(rc)

    def canBuild(self, card: 'Card') -> bool:
        cost = self.realGoldCost(card)
        return self.tokens[GOLD] >= cost[GOLD]

    def gotToManyTokens(self) -> bool:
        return sum(self.tokens) > MAX_NB_TOKENS

    def getAllVisible(self, board: 'Board') -> List[List['Card']]:
        return board.displayedCards + [self.reserved]

    def canReserve(self) -> bool:
        return len(self.reserved) < MAX_RESERVE
