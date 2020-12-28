class Move:
    """ Define what a move is
    """

    def __init__(self, actionType, action, tokensToRemove, characterSelected):
        self.actionType = actionType # can be BUILD, RESERVE, TAKE2, TAKE3
        self.action = action # if build or reserve : [[lvl][pos]], if take tokens : list of tokens
        self.tokensToRemove = tokensToRemove # tokens the player will remove at the end of his turn if he have too many of them
        self.character = characterSelected # the noble the player will take if he can take one

    def __repr__(self):
        return f"[{self.actionType} {self.action} {self.tokensToRemove} {self.character}]"

    def __eq__(self, other):
        return self.actionType == other.actionType and self.action == other.action and self.tokensToRemove == other.tokensToRemove and self.character == other.character

    def __ne__(self, other):
        return not self.__eq__(other)
