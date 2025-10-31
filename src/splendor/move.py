from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .cards import Card
    from .characters import Character

class Move:
    """ Define what a move is
    """

    def __init__(self, actionType: int, action: Union[List[int], 'Card', int], tokensToRemove: List[int], characterSelected: Optional['Character']) -> None:
        self.actionType: int = actionType # can be BUILD, RESERVE, TAKE2, TAKE3
        self.action: Union[List[int], 'Card', int] = action # if build or reserve from visible : the card, if take tokens : list of tokens, if reserve from top deck then the deck level (int)
        self.tokensToRemove: List[int] = tokensToRemove # tokens the player will remove at the end of his turn if he have too many of them
        self.character: Optional['Character'] = characterSelected # the noble the player will take if he can take one

    def __repr__(self) -> str:
        return f"[{self.actionType} {self.action} {self.tokensToRemove} {self.character}]"

    def __eq__(self, other: object) -> bool:
        return self.actionType == other.actionType and self.action == other.action and self.tokensToRemove == other.tokensToRemove and self.character == other.character

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
