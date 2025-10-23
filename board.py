# code implemented thanks to https://gist.github.com/kjlubick/8ea239ede6a026a61f4d
# I first coded all the game and then adapted it to fit to their code
# Code for ISMCTS Written by Peter Cowling, Edward Powley, Daniel Whitehouse (University of York, UK) September 2012 - August 2013.
# the rest by Maël Simon december 2020

from typing import List, Optional, Union, Any
from custom_operators import *
from constants import *
from random import shuffle
from player import *
from characters import *
from move import *
from cards import *
from copy import deepcopy
from itertools import permutations

class Board:
    """ A state of the game splendor
        visit to see full rulebook https://cdn.1j1ju.com/medias/7f/91/ba-splendor-rulebook.pdf
    """
    def __init__(self, nbPlayer: int, IA: List[Optional[str]], debug: bool = False) -> None:
        self.debug: bool = debug
        self.deckLVL1: List[Card] = [Card(c[0], c[1], c[2], c[3]) for c in DECK1]
        shuffle(self.deckLVL1)
        self.deckLVL2: List[Card] = [Card(c[0], c[1], c[2], c[3]) for c in DECK2]
        shuffle(self.deckLVL2)
        self.deckLVL3: List[Card] = [Card(c[0], c[1], c[2], c[3]) for c in DECK3]
        shuffle(self.deckLVL3)
        self.decks: List[List[Card]] = [self.deckLVL1, self.deckLVL2, self.deckLVL3]
        nbToken = self.getNbMaxTokens(nbPlayer)
        # color order : white blue green red black gold
        self.tokens: List[int] = [nbToken] * 5 + [5]
        self.characters: List[Character] = [Character(c[0], c[1]) for c in CHARACTERS]
        shuffle(self.characters)
        self.characters = self.characters[:nbPlayer + 1]
        self.displayedCards: List[List[Card]] = [[self.deckLVL1.pop(0) for i in range(0,4)], [self.deckLVL2.pop(0) for i in range(0,4)], [self.deckLVL3.pop(0) for i in range(0,4)]]
        for c in flatten(self.displayedCards):
            c.setVisible()
        self.players: List[Player] = [Player(str(i), IA[i]) for i in range(0,nbPlayer)]
        self.endGame: bool = False
        self.currentPlayer: int = 0
        self.nbTurn: int = 1
        self.isFinish: bool = False
    
    def getNbMaxTokens(self, nbPlayer: int) -> int:
       if nbPlayer == 2:
           return NB_TOKEN_2
       elif nbPlayer == 3:
           return NB_TOKEN_3
       else : return NB_TOKEN_4

    def clone(self) -> 'Board':
        nbPlayer = len(self.players)
        st = Board(nbPlayer, [None] * nbPlayer)
        st.decks = deepcopy(self.decks)
        st.tokens = deepcopy(self.tokens)
        st.characters = deepcopy(self.characters)
        st.displayedCards = deepcopy(self.displayedCards)
        st.players = deepcopy(self.players)
        st.endGame = self.endGame
        st.currentPlayer = self.currentPlayer
        st.nbTurn = self.nbTurn
        st.isFinish = self.isFinish
        return st

    def cloneAndRandomize(self, observer: Player) -> 'Board':
        """ Create a deep clone of this game state, ranfomizing any information not visible to the specified observer player
        """
        st = self.clone()
        # the observer can see all cards except those coming directly from the decks (decks themslef and reserved by other player coming from the decks)
        unseenCardsFromOtherPlayers = [player.getUnseenCards() for player in st.players if player != observer]
        unseenCards = st.decks
        for card in flatten(unseenCardsFromOtherPlayers):
            st.decks[card.lvl - 1].append(card)
        # shuffle all unseen cards
        for deck in unseenCards:
            shuffle(deck)
        # Deal unseen cards to players who have unseen reserved cards
        for player in st.players:
            if player != observer and player.haveUnseenCards():
                unseen = player.getUnseenCards()
                player.removeUnseenCards()
                for c in unseen:
                    # replace unseen cards by randomize unseend card from the same lvl
                    player.reserved.append(unseenCards[c.lvl - 1].pop(0))
        st.decks = unseenCards
        return st

    def getCurrentPlayer(self) -> Player:
        return self.players[self.currentPlayer]

    def getNextPlayer(self, currentPlayer: int) -> Player:
        nxt = (currentPlayer + 1) % len(self.players)
        return self.players[nxt]

    def nextPlayer(self) -> None:
        self.checkEndGame()
        self.currentPlayer = (self.currentPlayer + 1) % len(self.players)
        if self.currentPlayer == 0 and not self.endGame:
            self.nbTurn += 1
        if self.endGame and self.currentPlayer == 0:
            self.isFinish = True

    def doMove(self, move: Move) -> None:
        """ Update a state by carrying out the given move
        must update current player
        """
        if move.actionType == BUILD:
            self.build(move)
        elif move.actionType == RESERVE:
            self.reserve(move)
        elif move.actionType == TOKENS:
            self.takeTokens(move)
        else:
            print(f"Unknown action {move.actionType}")

        self.removeTooManyTokens(move)
        self.takeCharacter(move)
        self.nextPlayer()
        
    def getMoves(self) -> List[Move]:
        """ get all possible moves from this state
        """
        if self.isFinish: return []
        allTokens = self.getPossibleTokens()
        movesTokens = self.makeMovesTokens(allTokens)
        allBuild = self.getPossibleBuild()
        movesBuild = self.makeMovesBuild(allBuild)
        allReserve = self.getPossibleReserve()
        movesReserve = self.makeMovesReserve(allReserve)
        t = len(movesTokens)
        b = len(movesBuild)
        r = len(movesReserve)
        total = t + b + r
        # print(f"{t} {b} {r} {self.getCurrentPlayer().tokens} {sum(self.getCurrentPlayer().tokens)}")
        if sum(self.getCurrentPlayer().tokens) >= MAX_NB_TOKENS and movesBuild:
            movesTokens = []
        if movesTokens or movesBuild:
           movesReserve = []
        return movesTokens + movesBuild + movesReserve

    def getPossibleTokens(self) -> List[List[int]]:
        """ return all combinations of tokens that it's possible to take
        """
        # first create all combinations of tokens
        nbPossible = 5 - self.tokens[:5].count(0)
        all3_3 = set(permutations([1,1,1,0,0])) if nbPossible >= 3 else []
        all3_2 = set(permutations([1,1,0,0,0])) if nbPossible == 2 else []
        all3_1 = set(permutations([1,0,0,0,0])) if nbPossible == 1 else []
        all2 = set(permutations([2,0,0,0,0]))
        allComb3 = list(all3_3) + list(all3_2) + list(all3_1)
        # check validity of each combination
        valideComb3 = list(filter((lambda tokens: (all(t >= 0 for t in substract(self.tokens, tokens)))), map((lambda comb: list(comb) + [0]) ,allComb3)))
        valideComb2 = [comb + [0] for comb in map(list, all2) if self.tokens[comb.index(2)] >= NB_MINI_TOKEN_TAKE_2]
        return valideComb3 + valideComb2
        
    def makeMovesTokens(self, allTokens: List[List[int]]) -> List[Move]:
        """ create all moves possibles when choosing to take tokens
        """
        moves = []
        for comb in allTokens:
            tokensGiveAway = []
            tokensPlayerAfter = add(self.getCurrentPlayer().tokens, comb)
            nbToGive = sum(tokensPlayerAfter) - MAX_NB_TOKENS
            if nbToGive > 0:
                # you can't have more than 3 tokens to give away because in one turn you can't take more than 3 tokens
                if nbToGive == 1:
                    tokensGiveAway = list(set(permutations([1,0,0,0,0,0])))
                elif nbToGive == 2:
                    tokensGiveAway = list(set(permutations([1,1,0,0,0,0]))) + list(set(permutations([2,0,0,0,0,0])))
                else:
                    tokensGiveAway = list(set(permutations([1,1,1,0,0,0]))) + list(set(permutations([2,1,0,0,0,0]))) + list(set(permutations([3,0,0,0,0,0])))
                # check validity for each combination
                tokensGiveAway = list(map(list, tokensGiveAway))
                tokensGiveAway = list(filter((lambda lt: all(t >= 0 for t in substract(tokensPlayerAfter, lt))), tokensGiveAway))
                #print(tokensGiveAway)
                #print(tokensPlayerAfter)
                #raise Exception("stop")
            moves += [Move(TOKENS, comb, tga, None) for tga in tokensGiveAway] if tokensGiveAway else [Move(TOKENS, comb, NO_TOKENS, None)]
        return moves

    def getPossibleBuild(self) -> List[Card]:
        """ return position of all cards the player can build
        """
        player = self.getCurrentPlayer()
        allVisible = player.getAllVisible(self)
        build = []
        for i in range(0,len(allVisible)):
            for j in range(0, len(allVisible[i])):
                card = allVisible[i][j]
                if player.canBuild(card):
                    build.append(card)
        return build

    def makeMovesBuild(self, allBuild: List[Card]) -> List[Move]:
        """ create all moves possibles when choosing to  build
        """
        moves = []
        bonus = self.getCurrentPlayer().getTotalBonus()
        for card in allBuild:
            newBonus = bonus.copy()
            newBonus[card.bonus] += 1
            possible = list(filter(lambda c: all(color >=0 for color in substract(newBonus, c.cost)), self.characters))
            moves += [Move(BUILD, card, NO_TOKENS, p) for p in possible] if possible else [Move(BUILD, card, NO_TOKENS, None)]
        return moves
        
    def getPossibleReserve(self) -> List[Union[Card, int]]:
        """ return all combinations of reserve that it's possible to take
        """
        reserve: List[Union[Card, int]] = []
        if self.getCurrentPlayer().canReserve():
            for i in range(0,len(self.displayedCards)):
                # add visible cards
                for j in range(0, len(self.displayedCards[i])):
                    reserve.append(self.displayedCards[i][j])
                # if there's still a deck of this lvl
                if self.decks[i]:
                    reserve.append(i)
        return reserve

    def makeMovesReserve(self, allReserve: List[Union[Card, int]]) -> List[Move]:
        """ create all moves possibles when choosing to  reserve
        """
        tokensGiveAway = list(map(list, set(permutations([1,0,0,0,0,0]))))
        tokensGiveAway = list(filter((lambda lt: all(t >= 0 for t in substract(self.getCurrentPlayer().tokens, lt))), tokensGiveAway))

        toRemove = tokensGiveAway if sum(self.getCurrentPlayer().tokens) == MAX_NB_TOKENS and self.tokens[GOLD] else []
        moves = []
        for reserve in allReserve:
            if toRemove:
                for rm in toRemove:
                    moves += [Move(RESERVE, reserve, rm, None)]
            else:
                moves += [Move(RESERVE, reserve, NO_TOKENS, None)]
        return moves
        
    def getVictorious(self, verbose: bool = False) -> int:
        vp = [player.getVictoryPoints() for player in self.players]
        indices = [i for i, p in enumerate(vp) if p == max(vp)]
        if len(indices) == 1:
            winner = indices[0]
        else:
            if verbose: print("some players have the same amout of victory points. the winner is the one with less cards")
            nbCard = [len(self.players[i].built) for i in indices]
            winner = indices[nbCard.index(min(nbCard))]
        if verbose: print("player " + self.players[winner].name + " won")
        return winner

    def getResult(self, player: Player) -> int:
        """ return 1 if the player is the one who won this game, 0 otherwise
        """
        # print(f"get result, turn n{self.nbTurn}")
        score = 1 if player == self.players[self.getVictorious()] else 0
        return score

    def checkEndGame(self, verbose: bool = False) -> None:
        player = self.getCurrentPlayer()
        if player.getVictoryPoints() >= VP_GOAL:
           if verbose: print(f"{bcolors.RED + bcolors.BOLD}WARNING it's the last turn ! {bcolors.RESET}")
           self.endGame = True

    def getCard(self, move: Move) -> Card:
        if move.actionType == RESERVE and move.action in TOP_DECK: # reserve topdeck
            return self.decks[move.action][0]
        else:
            return move.action

    def build(self, move: Move) -> None:
        """ current player build a card
        """
        card = self.getCard(move)
        player = self.getCurrentPlayer()
        rgc = player.realGoldCost(card)
        player.tokens = substract(player.tokens, rgc)
        self.tokens = add(self.tokens, rgc)
        if card in player.reserved:
            player.reserved.remove(card)
            card.setVisible()
        else:
            self.removeCard(move)
        player.built.append(card)

    def reserve(self, move: Move) -> None:
        card = self.getCard(move)
        self.removeCard(move)
        self.getCurrentPlayer().reserved.append(card)
        if self.tokens[GOLD]: self.takeTokens(Move(move.actionType, TAKEONEGOLD, move.tokensToRemove, move.character))

    def takeTokens(self, move: Move) -> None:
        player = self.getCurrentPlayer()
        player.tokens = add(player.tokens, move.action)
        self.tokens = substract(self.tokens, move.action)
        if any(t < 0 for t in self.tokens):
            raise Exception("negatif tokens")

    def removeTooManyTokens(self, move: Move) -> None:
        player = self.getCurrentPlayer()
        player.tokens = substract(player.tokens, move.tokensToRemove)
        self.tokens = add(self.tokens, move.tokensToRemove)
        if any(t < 0 for t in player.tokens):
            raise Exception("negatif tokens")

    def takeCharacter(self, move: Move) -> None:
        if move.character:
            self.getCurrentPlayer().characters.append(move.character)
            self.characters.remove(move.character)

    def removeCard(self, move: Move) -> None:
        if move.action in TOP_DECK: #top deck
            del self.decks[move.action][0]
        else:
            #card = next(filter((lambda c: c == move.action), self.displayedCards[move.action.lvl - 1]))
            card = move.action
            self.displayedCards[card.lvl - 1].remove(card)
            if len(self.decks[card.lvl - 1]) > 0:
                newCard = self.decks[card.lvl - 1].pop(0)
                newCard.setVisible()
                self.displayedCards[card.lvl - 1].append(newCard)

    def show(self) -> None:
        print("===============================================================")
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

    def getShowTokens(self) -> List[str]:
        return [f"{getColor(color)}{qtt}" for color, qtt in enumerate(self.tokens)]

    def getState(self) -> List[Any]:
        return [self.nbTurn, self.currentPlayer, self.tokens.copy(), self.displayedCards[0].copy(), self.displayedCards[1].copy(), self.displayedCards[2].copy(), self.characters.copy()]

    def getPlayerState(self, playerNumber: int) -> List[int]:
        player = self.players[playerNumber]
        return player.tokens.copy()
